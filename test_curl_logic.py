import math
import unittest

import numpy as np

from open_camera import ArmPose, CurlSession


def arm_with_angle(angle_degrees, elbow=(200.0, 200.0), confidence=0.99):
    theta = math.radians(angle_degrees)
    elbow = np.array(elbow, dtype=float)
    shoulder = elbow + np.array([0.0, -100.0])
    wrist = elbow + np.array([100.0 * math.sin(theta), -100.0 * math.cos(theta)])
    return ArmPose(
        side="right",
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
        confidence=confidence,
        angle=angle_degrees,
        upper_arm_length=100.0,
    )


class CurlSessionTest(unittest.TestCase):
    def test_counts_clean_curl(self):
        session = CurlSession()
        session.start()

        session.update(arm_with_angle(165))
        session.update(arm_with_angle(120))
        session.rep_started_at -= 1.2
        session.update(arm_with_angle(60))
        session.update(arm_with_angle(90))
        session.update(arm_with_angle(165))

        self.assertEqual(session.clean_reps, 1)
        self.assertEqual(session.total_attempts, 1)
        self.assertEqual(session.reps[0].issues, [])

    def test_rejects_partial_curl(self):
        session = CurlSession()
        session.start()

        session.update(arm_with_angle(165))
        session.update(arm_with_angle(120))
        session.rep_started_at -= 1.2
        session.update(arm_with_angle(90))
        session.update(arm_with_angle(165))

        self.assertEqual(session.clean_reps, 0)
        self.assertEqual(session.total_attempts, 1)
        self.assertIn("partial_curl", session.reps[0].issues)

    def test_rejects_fast_curl(self):
        session = CurlSession()
        session.start()

        session.update(arm_with_angle(165))
        session.update(arm_with_angle(120))
        session.update(arm_with_angle(60))
        session.update(arm_with_angle(90))
        session.update(arm_with_angle(165))

        self.assertEqual(session.clean_reps, 0)
        self.assertEqual(session.total_attempts, 1)
        self.assertIn("too_fast", session.reps[0].issues)

    def test_tracks_perception_warnings(self):
        session = CurlSession()
        session.start()

        session.update(None)
        session.update(arm_with_angle(165, confidence=0.1))

        summary = session.summary()
        self.assertEqual(summary["perception_warnings"]["arm_not_visible"], 1)
        self.assertEqual(summary["perception_warnings"]["low_confidence"], 1)


if __name__ == "__main__":
    unittest.main()
