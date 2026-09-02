import unittest

from PIL import Image

from src.dataset_loader import (
    JetRacerDataset,
    smooth_steering_labels,
    session_split,
)


class RecoveryAugmentationTests(unittest.TestCase):
    def test_right_shift_adds_right_steering_correction(self):
        dataset = JetRacerDataset(
            [],
            steering_correction_gain=2.0,
        )
        image = Image.new('RGB', (100, 50), color=(20, 30, 40))

        shifted, steering = dataset._apply_recovery_shift(
            image, steering=0.0, shift_pixels=10
        )

        self.assertEqual(shifted.size, image.size)
        self.assertAlmostEqual(steering, 0.20, places=6)

    def test_corrected_target_is_clipped(self):
        dataset = JetRacerDataset(
            [],
            steering_correction_gain=2.0,
        )
        image = Image.new('RGB', (100, 50), color=(20, 30, 40))

        _, steering = dataset._apply_recovery_shift(
            image, steering=0.95, shift_pixels=20
        )

        self.assertEqual(steering, 1.0)


class SessionSplitTests(unittest.TestCase):
    def test_sessions_do_not_leak_between_train_and_validation(self):
        entries = []
        for session, count in [('lap_a', 30), ('lap_b', 20), ('lap_c', 50)]:
            for index in range(count):
                entries.append({
                    'session': session,
                    'steering': 0.0,
                    'image_path': '{}_{}.jpg'.format(session, index),
                })

        train_entries, val_entries = session_split(
            entries, val_ratio=0.2, seed=42
        )
        train_sessions = {entry['session'] for entry in train_entries}
        val_sessions = {entry['session'] for entry in val_entries}

        self.assertTrue(train_sessions)
        self.assertTrue(val_sessions)
        self.assertTrue(train_sessions.isdisjoint(val_sessions))
        self.assertEqual(len(train_entries) + len(val_entries), len(entries))


class LabelCleaningTests(unittest.TestCase):
    def test_isolated_joystick_release_is_removed_without_length_change(self):
        cleaned = smooth_steering_labels([0.5, 0.5, 0.0, 0.5, 0.5])
        self.assertEqual(len(cleaned), 5)
        self.assertGreater(cleaned[2], 0.45)


if __name__ == '__main__':
    unittest.main()
