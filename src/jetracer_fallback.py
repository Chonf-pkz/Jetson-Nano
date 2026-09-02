"""Local JetRacer motor driver compatible with NVIDIA's reference package.

This keeps the project runnable when the small ``jetracer`` Python package is
missing but Adafruit ServoKit is already installed on the Jetson image.
"""

import traitlets

try:
    from adafruit_servokit import ServoKit
except ImportError as error:
    raise ImportError(
        'Thiếu cả jetracer và adafruit_servokit. Cài ServoKit trước bằng '
        '"sudo python3 -m pip install adafruit-circuitpython-servokit".'
    ) from error


class Racecar(traitlets.HasTraits):
    steering = traitlets.Float(default_value=0.0)
    throttle = traitlets.Float(default_value=0.0)

    @traitlets.validate('steering')
    def _clip_steering(self, proposal):
        return max(-1.0, min(1.0, float(proposal['value'])))

    @traitlets.validate('throttle')
    def _clip_throttle(self, proposal):
        return max(-1.0, min(1.0, float(proposal['value'])))


class NvidiaRacecar(Racecar):
    i2c_address = traitlets.Integer(default_value=0x40)
    steering_gain = traitlets.Float(default_value=-0.65)
    steering_offset = traitlets.Float(default_value=0.0)
    steering_channel = traitlets.Integer(default_value=0)
    throttle_gain = traitlets.Float(default_value=0.8)
    throttle_channel = traitlets.Integer(default_value=1)

    def __init__(self, *args, **kwargs):
        super(NvidiaRacecar, self).__init__(*args, **kwargs)
        self.kit = ServoKit(channels=16, address=self.i2c_address)
        self.steering_motor = self.kit.continuous_servo[
            self.steering_channel
        ]
        self.throttle_motor = self.kit.continuous_servo[
            self.throttle_channel
        ]
        self.steering = 0.0
        self.throttle = 0.0

    @traitlets.observe('steering')
    def _on_steering(self, change):
        if hasattr(self, 'steering_motor'):
            self.steering_motor.throttle = (
                change['new'] * self.steering_gain + self.steering_offset
            )

    @traitlets.observe('throttle')
    def _on_throttle(self, change):
        if hasattr(self, 'throttle_motor'):
            self.throttle_motor.throttle = (
                change['new'] * self.throttle_gain
            )
