"""Adaptive steering and speed controller for JetRacer.

The neural network predicts the direction of the road.  This controller turns
that prediction into motor commands while accounting for three things the
single-frame model cannot handle well by itself:

* anticipate a corner from the fast/slow prediction trend;
* reduce throttle as curvature or prediction instability increases;
* accelerate slowly on a straight, but decelerate quickly before a corner.

The implementation intentionally has no PyTorch dependency so it can also be
unit-tested and tuned directly on a Jetson Nano.
"""

from collections import deque, namedtuple
import math
import statistics
import time


def _clip(value, lower, upper):
    return max(lower, min(upper, float(value)))


DriveCommand = namedtuple(
    'DriveCommand',
    [
        'steering',
        'throttle',
        'target_throttle',
        'curve_demand',
        'instability',
    ],
)
DriveCommand.__doc__ = "One controller output plus telemetry useful for tuning."


class AdaptiveDriveController:
    """Convert raw steering predictions into adaptive steering and throttle.

    ``max_throttle`` is the fastest command allowed on a stable straight.
    ``min_throttle`` is the command used for a very tight or uncertain corner.
    Values are JetRacer normalized motor commands, not measured vehicle speed.
    """

    def __init__(
        self,
        min_throttle=0.18,
        max_throttle=0.45,
        max_steering=0.90,
        steering_gain=1.15,
        steering_exponent=0.85,
        high_speed_steering_gain=0.10,
        dead_zone=0.03,
        anticipation_gain=0.55,
        curve_full_scale=0.75,
        curve_exponent=1.35,
        straight_time_constant=0.14,
        corner_time_constant=0.045,
        steering_rate_limit=6.0,
        acceleration_rate=0.18,
        deceleration_rate=1.20,
        instability_weight=0.30,
        confidence_weight=0.55,
        history_size=8,
    ):
        if not 0.0 <= min_throttle <= max_throttle <= 1.0:
            raise ValueError(
                "Expected 0 <= min_throttle <= max_throttle <= 1"
            )
        if not 0.0 < max_steering <= 1.0:
            raise ValueError("max_steering must be in (0, 1]")
        if steering_gain <= 0.0 or steering_exponent <= 0.0:
            raise ValueError("Steering gain and exponent must be positive")
        if curve_full_scale <= 0.0:
            raise ValueError("curve_full_scale must be positive")
        if acceleration_rate <= 0.0 or deceleration_rate <= 0.0:
            raise ValueError("Throttle rates must be positive")
        if history_size < 3:
            raise ValueError("history_size must be at least 3")

        self.min_throttle = float(min_throttle)
        self.max_throttle = float(max_throttle)
        self.max_steering = float(max_steering)
        self.steering_gain = float(steering_gain)
        self.steering_exponent = float(steering_exponent)
        self.high_speed_steering_gain = float(high_speed_steering_gain)
        self.dead_zone = float(dead_zone)
        self.anticipation_gain = float(anticipation_gain)
        self.curve_full_scale = float(curve_full_scale)
        self.curve_exponent = float(curve_exponent)
        self.straight_time_constant = float(straight_time_constant)
        self.corner_time_constant = float(corner_time_constant)
        self.steering_rate_limit = float(steering_rate_limit)
        self.acceleration_rate = float(acceleration_rate)
        self.deceleration_rate = float(deceleration_rate)
        self.instability_weight = float(instability_weight)
        self.confidence_weight = float(confidence_weight)
        self._history = deque(maxlen=int(history_size))

        self.reset()

    def reset(self, initial_throttle=None):
        """Clear temporal state, for example after an emergency stop."""
        self._fast_prediction = 0.0
        self._slow_prediction = 0.0
        self._steering = 0.0
        if initial_throttle is None:
            initial_throttle = self.min_throttle
        self._throttle = _clip(
            initial_throttle, self.min_throttle, self.max_throttle
        )
        self._last_time = None
        self._initialized = False
        self._history.clear()

    def set_throttle_limits(self, min_throttle, max_throttle):
        """Update safe speed limits while keeping the controller state valid."""
        if not 0.0 <= min_throttle <= max_throttle <= 1.0:
            raise ValueError(
                "Expected 0 <= min_throttle <= max_throttle <= 1"
            )
        self.min_throttle = float(min_throttle)
        self.max_throttle = float(max_throttle)
        self._throttle = _clip(
            self._throttle, self.min_throttle, self.max_throttle
        )

    @property
    def throttle(self):
        return self._throttle

    @property
    def speed_fraction(self):
        span = self.max_throttle - self.min_throttle
        if span <= 1e-6:
            return 0.0
        return _clip(
            (self._throttle - self.min_throttle) / span, 0.0, 1.0
        )

    @staticmethod
    def _ema(previous, value, dt, time_constant):
        time_constant = max(float(time_constant), 1e-4)
        alpha = 1.0 - math.exp(-dt / time_constant)
        return previous + alpha * (value - previous)

    def _prediction_instability(self):
        if len(self._history) < 3:
            return 0.0

        values = list(self._history)
        # Ordinary steering corrections on the real track often vary by
        # 0.20-0.30.  Reserve high instability for sustained, larger jumps.
        spread = min(1.0, statistics.pstdev(values) / 0.35)

        signs = []
        for value in values:
            if abs(value) >= 0.08:
                signs.append(1 if value > 0.0 else -1)
        reversals = sum(a != b for a, b in zip(signs, signs[1:]))
        reversal_ratio = reversals / max(len(signs) - 1, 1)

        return _clip(0.80 * spread + 0.20 * reversal_ratio, 0.0, 1.0)

    def update(
        self,
        raw_steering,
        dt=None,
        confidence=None,
        external_risk=0.0,
    ):
        """Produce the next steering/throttle command.

        Args:
            raw_steering: Model output in ``[-1, 1]``.
            dt: Seconds since the previous frame.  If omitted, a monotonic
                clock is used.  The value is bounded to reject camera stalls.
            confidence: Optional lane confidence in ``[0, 1]``.  The current
                steering-only model has no confidence output, so this may be
                left as ``None``; temporal instability remains active.
            external_risk: Optional geometric curve/edge risk in ``[0, 1]``.
        """
        now = time.monotonic()
        if dt is None:
            dt = 1.0 / 30.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        dt = _clip(dt, 1.0 / 120.0, 0.20)

        prediction = _clip(raw_steering, -1.0, 1.0)
        if abs(prediction) <= self.dead_zone:
            prediction = 0.0
        else:
            prediction = math.copysign(
                (abs(prediction) - self.dead_zone) / (1.0 - self.dead_zone),
                prediction,
            )

        if not self._initialized:
            self._fast_prediction = prediction
            self._slow_prediction = prediction
            self._initialized = True
        else:
            self._fast_prediction = self._ema(
                self._fast_prediction, prediction, dt, 0.070
            )
            self._slow_prediction = self._ema(
                self._slow_prediction, prediction, dt, 0.30
            )

        anticipated = self._fast_prediction + self.anticipation_gain * (
            self._fast_prediction - self._slow_prediction
        )
        anticipated = _clip(anticipated, -1.0, 1.0)
        self._history.append(anticipated)

        curve_demand = _clip(
            abs(anticipated) / self.curve_full_scale, 0.0, 1.0
        )
        instability = self._prediction_instability()

        shaped = math.copysign(
            abs(anticipated) ** self.steering_exponent, anticipated
        )
        speed_fraction = self._throttle / max(self.max_throttle, 1e-6)
        speed_compensation = (
            1.0
            + self.high_speed_steering_gain * speed_fraction * curve_demand
        )
        desired_steering = _clip(
            shaped * self.steering_gain * speed_compensation,
            -self.max_steering,
            self.max_steering,
        )

        steering_tau = (
            self.straight_time_constant * (1.0 - curve_demand)
            + self.corner_time_constant * curve_demand
        )
        filtered_steering = self._ema(
            self._steering, desired_steering, dt, steering_tau
        )
        max_steering_step = self.steering_rate_limit * dt
        self._steering += _clip(
            filtered_steering - self._steering,
            -max_steering_step,
            max_steering_step,
        )
        self._steering = _clip(
            self._steering, -self.max_steering, self.max_steering
        )

        confidence_risk = 0.0
        if confidence is not None:
            confidence_risk = 1.0 - _clip(confidence, 0.0, 1.0)

        risk = (
            curve_demand ** self.curve_exponent
            + self.instability_weight * instability
            + self.confidence_weight * confidence_risk
        )
        risk = max(risk, _clip(external_risk, 0.0, 1.0))
        risk = _clip(risk, 0.0, 1.0)
        throttle_range = self.max_throttle - self.min_throttle
        target_throttle = self.max_throttle - throttle_range * risk

        throttle_rate = (
            self.deceleration_rate
            if target_throttle < self._throttle
            else self.acceleration_rate
        )
        max_throttle_step = throttle_rate * dt
        self._throttle += _clip(
            target_throttle - self._throttle,
            -max_throttle_step,
            max_throttle_step,
        )
        self._throttle = _clip(
            self._throttle, self.min_throttle, self.max_throttle
        )

        return DriveCommand(
            steering=self._steering,
            throttle=self._throttle,
            target_throttle=target_throttle,
            curve_demand=curve_demand,
            instability=instability,
        )
