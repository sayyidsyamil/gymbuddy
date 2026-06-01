#!/usr/bin/env python3
"""Form analysis for bicep curl and lateral raise.

Publishes FormStatus on every skeleton frame.
  elbow_angle field carries:
    bicep_curl    — elbow flexion angle  (155° = arm straight, 90° = full curl)
    lateral_raise — shoulder abduction angle (0° = arm at side, 90° = horizontal)
"""

import math
import threading

import numpy as np
import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import FormStatus, Skeleton

CURL_ARMS = {
    "left":  {"shoulder": 5, "elbow": 7, "wrist": 9},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10},
}
LATERAL_ARMS = {
    "left":  {"shoulder": 5, "elbow": 7, "hip": 11},
    "right": {"shoulder": 6, "elbow": 8, "hip": 12},
}

# Bicep curl thresholds — must stay in sync with rep_counter_node.py
CURL_EXTENDED    = 145.0
CURL_TOP         = 95.0
CURL_PARTIAL     = 115.0

# Lateral raise thresholds — shoulder abduction angle
LAT_DOWN         = 30.0   # arm at side
LAT_PARTIAL      = 55.0   # partial raise
LAT_TOP          = 75.0   # arm at shoulder height

# Bicep curl only needs shoulder+elbow+wrist visible — keep the bar low so
# close-up framing (where hips often leave the frame) still works.
MIN_VISIBILITY_CURL    = 0.20
# Lateral raise requires the hip as the torso reference, so we keep this higher.
MIN_VISIBILITY_LATERAL = 0.35


def angle_between(a, b, c):
    """Angle in degrees at vertex b formed by rays b→a and b→c."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 180.0
    cosine = np.dot(ba, bc) / denom
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _pt(lm, w, h):
    return np.array([lm.x * w, lm.y * h])


class FormAnalysisNode:
    def __init__(self):
        self.extended_angle    = float(rospy.get_param("~extended_angle",    CURL_EXTENDED))
        self.top_angle         = float(rospy.get_param("~top_angle",         CURL_TOP))
        self.partial_top_angle = float(rospy.get_param("~partial_top_angle", CURL_PARTIAL))
        self.lat_down_angle    = float(rospy.get_param("~lat_down_angle",    LAT_DOWN))
        self.lat_partial_angle = float(rospy.get_param("~lat_partial_angle", LAT_PARTIAL))
        self.lat_top_angle     = float(rospy.get_param("~lat_top_angle",     LAT_TOP))
        self.min_visibility_curl    = float(rospy.get_param("~min_visibility_curl",    MIN_VISIBILITY_CURL))
        self.min_visibility_lateral = float(rospy.get_param("~min_visibility_lateral", MIN_VISIBILITY_LATERAL))

        self._exercise = "bicep_curl"
        self._lock = threading.Lock()

        self.pub = rospy.Publisher("/form_status", FormStatus, queue_size=4)
        rospy.Subscriber("/skeleton_data",   Skeleton, self.on_skeleton, queue_size=2)
        rospy.Subscriber("/active_exercise", String,   self.on_exercise, queue_size=1)
        rospy.loginfo("form_analysis_node ready (exercise=%s)", self._exercise)

    def on_exercise(self, msg: String):
        with self._lock:
            self._exercise = msg.data
        rospy.loginfo("form_analysis: switched to %s", msg.data)

    # ------------------------------------------------------------------ #
    # Bicep curl                                                           #
    # ------------------------------------------------------------------ #

    def _get_curl_arm(self, skel: Skeleton, side: str):
        w, h = skel.image_width, skel.image_height
        idx = CURL_ARMS[side]
        s, e, wt = skel.landmarks[idx["shoulder"]], skel.landmarks[idx["elbow"]], skel.landmarks[idx["wrist"]]
        vis = (s.visibility + e.visibility + wt.visibility) / 3.0
        if vis < self.min_visibility_curl:
            return None
        angle = angle_between(_pt(s, w, h), _pt(e, w, h), _pt(wt, w, h))
        return {"side": side, "angle": angle, "confidence": vis}

    def _curl_status(self, angle: float) -> tuple:
        if angle <= self.top_angle:
            return "depth_reached", "elbow at top of curl"
        elif angle <= self.partial_top_angle:
            return "near_top", "almost at full curl"
        elif angle >= self.extended_angle:
            return "fully_extended", "ready position"
        else:
            return "mid_range", "in motion"

    def _analyze_bicep_curl(self, skel: Skeleton) -> FormStatus:
        out = FormStatus()
        out.header = skel.header

        left  = self._get_curl_arm(skel, "left")
        right = self._get_curl_arm(skel, "right")

        if left is None and right is None:
            out.status = "arm_not_visible"
            out.detail = "both arms below visibility threshold"
            out.elbow_angle       = float("nan")
            out.confidence        = 0.0
            out.right_elbow_angle = float("nan")
            out.right_confidence  = 0.0
            return out

        # left arm (primary angle field)
        if left is not None:
            l_status, l_detail = self._curl_status(left["angle"])
            out.elbow_angle = float(left["angle"])
            out.confidence  = float(left["confidence"])
        else:
            l_status = "arm_not_visible"
            l_detail = "not visible"
            out.elbow_angle = float("nan")
            out.confidence  = 0.0

        # right arm
        if right is not None:
            r_status, r_detail = self._curl_status(right["angle"])
            out.right_elbow_angle = float(right["angle"])
            out.right_confidence  = float(right["confidence"])
        else:
            r_status = "arm_not_visible"
            r_detail = "not visible"
            out.right_elbow_angle = float("nan")
            out.right_confidence  = 0.0

        # overall status: use the arm that's furthest from the top (worst case)
        if left is not None and right is not None:
            worst = left if left["angle"] > right["angle"] else right
            out.status = self._curl_status(worst["angle"])[0]
            out.detail = f"L:{left['angle']:.0f}deg R:{right['angle']:.0f}deg"
        elif left is not None:
            out.status = l_status
            out.detail = f"left: {l_detail} (right not visible)"
        else:
            out.status = r_status
            out.detail = f"right: {r_detail} (left not visible)"

        return out

    # ------------------------------------------------------------------ #
    # Lateral raise                                                        #
    # ------------------------------------------------------------------ #

    def _get_lateral_arm(self, skel: Skeleton, side: str):
        if len(skel.landmarks) < 13:
            return None
        w, h = skel.image_width, skel.image_height
        idx = LATERAL_ARMS[side]
        s, e, hip = skel.landmarks[idx["shoulder"]], skel.landmarks[idx["elbow"]], skel.landmarks[idx["hip"]]
        vis = (s.visibility + e.visibility + hip.visibility) / 3.0
        if vis < self.min_visibility_lateral:
            return None
        sp, ep, hp = _pt(s, w, h), _pt(e, w, h), _pt(hip, w, h)
        abduction = angle_between(hp, sp, ep)
        return {"side": side, "angle": abduction, "confidence": vis}

    def _lateral_status(self, angle: float) -> tuple:
        if angle <= self.lat_down_angle:
            return "arm_at_side",  "ready position"
        elif angle >= self.lat_top_angle:
            return "at_top",       "arm at shoulder height"
        elif angle >= self.lat_partial_angle:
            return "near_top",     "almost at shoulder height"
        else:
            return "mid_range",    "raising"

    def _analyze_lateral_raise(self, skel: Skeleton) -> FormStatus:
        out = FormStatus()
        out.header = skel.header

        left  = self._get_lateral_arm(skel, "left")
        right = self._get_lateral_arm(skel, "right")

        if left is None and right is None:
            out.status = "arm_not_visible"
            out.detail = "both arms below visibility threshold"
            out.elbow_angle       = float("nan")
            out.confidence        = 0.0
            out.right_elbow_angle = float("nan")
            out.right_confidence  = 0.0
            return out

        # left arm (primary angle field)
        if left is not None:
            l_status, l_detail = self._lateral_status(left["angle"])
            out.elbow_angle = float(left["angle"])
            out.confidence  = float(left["confidence"])
        else:
            l_status = "arm_not_visible"
            l_detail = "not visible"
            out.elbow_angle = float("nan")
            out.confidence  = 0.0

        # right arm
        if right is not None:
            r_status, r_detail = self._lateral_status(right["angle"])
            out.right_elbow_angle = float(right["angle"])
            out.right_confidence  = float(right["confidence"])
        else:
            r_status = "arm_not_visible"
            r_detail = "not visible"
            out.right_elbow_angle = float("nan")
            out.right_confidence  = 0.0

        # overall status: use the arm lowest (furthest from top, worst case)
        if left is not None and right is not None:
            worst = left if left["angle"] < right["angle"] else right
            out.status = self._lateral_status(worst["angle"])[0]
            out.detail = f"L:{left['angle']:.0f}deg R:{right['angle']:.0f}deg"
        elif left is not None:
            out.status = l_status
            out.detail = f"left: {l_detail} (right not visible)"
        else:
            out.status = r_status
            out.detail = f"right: {r_detail} (left not visible)"

        out.elbow_angle = out.elbow_angle   # shoulder abduction angle (left)
        return out

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    def on_skeleton(self, msg: Skeleton):
        if not msg.landmarks or len(msg.landmarks) < 17:
            return
        with self._lock:
            exercise = self._exercise
        if exercise == "lateral_raise":
            self.pub.publish(self._analyze_lateral_raise(msg))
        else:
            self.pub.publish(self._analyze_bicep_curl(msg))


def main():
    rospy.init_node("form_analysis_node")
    FormAnalysisNode()
    rospy.spin()


if __name__ == "__main__":
    main()
