#!/usr/bin/env python3
"""Computes elbow angle for the most-visible arm and publishes form status events."""

import math

import numpy as np
import rospy

from gymbuddy_ros.msg import FormStatus, Skeleton

# MediaPipe Pose landmark indices
LEFT_ARM = {"name": "left", "shoulder": 11, "elbow": 13, "wrist": 15}
RIGHT_ARM = {"name": "right", "shoulder": 12, "elbow": 14, "wrist": 16}

EXTENDED_ANGLE = 155.0
TOP_ANGLE = 70.0
PARTIAL_TOP_ANGLE = 100.0
MIN_VISIBILITY = 0.5


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 180.0
    cosine = np.dot(ba, bc) / denom
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


class FormAnalysisNode:
    def __init__(self):
        self.pub = rospy.Publisher("/form_status", FormStatus, queue_size=4)
        rospy.Subscriber("/skeleton_data", Skeleton, self.on_skeleton, queue_size=2)
        self.last_status = None
        rospy.loginfo("form_analysis_node ready")

    def select_arm(self, skel: Skeleton):
        if not skel.landmarks or len(skel.landmarks) < 17:
            return None

        candidates = []
        for arm in (LEFT_ARM, RIGHT_ARM):
            s = skel.landmarks[arm["shoulder"]]
            e = skel.landmarks[arm["elbow"]]
            w = skel.landmarks[arm["wrist"]]
            visibility = (s.visibility + e.visibility + w.visibility) / 3.0
            if visibility < MIN_VISIBILITY:
                continue
            shoulder = np.array([s.x * skel.image_width, s.y * skel.image_height])
            elbow = np.array([e.x * skel.image_width, e.y * skel.image_height])
            wrist = np.array([w.x * skel.image_width, w.y * skel.image_height])
            candidates.append({
                "name": arm["name"],
                "angle": angle_between(shoulder, elbow, wrist),
                "confidence": visibility,
            })
        if not candidates:
            return None
        return max(candidates, key=lambda a: a["confidence"])

    def classify(self, angle: float):
        if angle <= TOP_ANGLE:
            return "depth_reached", "elbow at top of curl"
        if angle <= PARTIAL_TOP_ANGLE:
            return "near_top", "almost at full curl"
        if angle >= EXTENDED_ANGLE:
            return "fully_extended", "ready position"
        return "mid_range", "in motion"

    def on_skeleton(self, msg: Skeleton):
        arm = self.select_arm(msg)
        out = FormStatus()
        out.header = msg.header

        if arm is None:
            out.status = "arm_not_visible"
            out.detail = "shoulder/elbow/wrist below visibility threshold"
            out.elbow_angle = float("nan")
            out.confidence = 0.0
        else:
            status, detail = self.classify(arm["angle"])
            out.status = status
            out.detail = f"{arm['name']}: {detail}"
            out.elbow_angle = float(arm["angle"])
            out.confidence = float(arm["confidence"])

        # publish on every skeleton frame so downstream nodes get fresh angles
        self.pub.publish(out)


def main():
    rospy.init_node("form_analysis_node")
    FormAnalysisNode()
    rospy.spin()


if __name__ == "__main__":
    main()
