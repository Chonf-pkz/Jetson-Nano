import unittest

from src.adaptive_controller import AdaptiveDriveController


class AdaptiveDriveControllerTests(unittest.TestCase):
    def make_controller(self):
        return AdaptiveDriveController(
            min_throttle=0.20,
            max_throttle=0.60,
            max_steering=0.90,
            acceleration_rate=0.20,
            deceleration_rate=1.00,
        )

    def test_stable_straight_accelerates_to_maximum(self):
        controller = self.make_controller()

        for _ in range(100):
            command = controller.update(0.0, dt=0.05)

        self.assertAlmostEqual(command.steering, 0.0, places=6)
        self.assertAlmostEqual(command.throttle, 0.60, places=6)
        self.assertAlmostEqual(command.target_throttle, 0.60, places=6)

    def test_tight_corner_decelerates_to_minimum(self):
        controller = self.make_controller()
        for _ in range(100):
            controller.update(0.0, dt=0.05)

        first_corner = controller.update(0.95, dt=0.05)
        self.assertLess(first_corner.throttle, 0.60)

        for _ in range(30):
            command = controller.update(0.95, dt=0.05)

        self.assertAlmostEqual(command.throttle, 0.20, places=4)
        self.assertGreater(command.curve_demand, 0.95)
        self.assertGreater(command.steering, 0.80)

    def test_noisy_direction_changes_raise_instability(self):
        controller = self.make_controller()

        values = [0.65, -0.65] * 8
        for value in values:
            command = controller.update(value, dt=0.04)

        self.assertGreater(command.instability, 0.45)
        self.assertLess(command.target_throttle, controller.max_throttle)

    def test_low_confidence_reduces_target_speed(self):
        high_confidence = self.make_controller().update(
            0.0, dt=0.05, confidence=1.0
        )
        low_confidence = self.make_controller().update(
            0.0, dt=0.05, confidence=0.0
        )

        self.assertLess(
            low_confidence.target_throttle,
            high_confidence.target_throttle,
        )

    def test_steering_is_bounded(self):
        controller = self.make_controller()

        for _ in range(50):
            command = controller.update(5.0, dt=0.05)

        self.assertLessEqual(abs(command.steering), 0.90)

    def test_external_lane_risk_forces_corner_speed(self):
        controller = self.make_controller()

        command = controller.update(
            0.0, dt=0.05, confidence=1.0, external_risk=1.0
        )

        self.assertAlmostEqual(command.target_throttle, 0.20, places=6)


if __name__ == '__main__':
    unittest.main()
