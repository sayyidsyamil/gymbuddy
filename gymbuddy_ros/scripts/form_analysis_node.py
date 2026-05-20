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

    def _best_curl_arm(self, skel: Skeleton):
        w, h = skel.image_width, skel.image_height
        candidates = []
        for side, idx in CURL_ARMS.items():
            s, e, wt = skel.landmarks[idx["shoulder"]], skel.landmarks[idx["elbow"]], skel.landmarks[idx["wrist"]]
            vis = (s.visibility + e.visibility + wt.visibility) / 3.0
            if vis < self.min_visibility_curl:
                continue
            angle = angle_between(_pt(s, w, h), _pt(e, w, h), _pt(wt, w, h))
            candidates.append({"side": side, "angle": angle, "confidence": vis,
                                "shoulder": _pt(s, w, h), "elbow": _pt(e, w, h), "wrist": _pt(wt, w, h)})
        return max(candidates, key=lambda a: a["confidence"]) if candidates else None

    def _analyze_bicep_curl(self, skel: Skeleton) -> FormStatus:
        out = FormStatus()
        out.header = skel.header
        arm = self._best_curl_arm(skel)
        if arm is None:
            out.status = "arm_not_visible"
            out.detail = "shoulder/elbow/wrist below visibility threshold"
            out.elbow_angle = float("nan")
            out.confidence = 0.0
            return out

        angle = arm["angle"]
        if angle <= self.top_angle:
            status, detail = "depth_reached", "elbow at top of curl"
        elif angle <= self.partial_top_angle:
            status, detail = "near_top", "almost at full curl"
        elif angle >= self.extended_angle:
            status, detail = "fully_extended", "ready position"
        else:
            status, detail = "mid_range", "in motion"

        out.status    = status
        out.detail    = f"{arm['side']}: {detail}"
        out.elbow_angle = float(angle)
        out.confidence  = float(arm["confidence"])
        return out

    # ------------------------------------------------------------------ #
    # Lateral raise                                                        #
    # ------------------------------------------------------------------ #

    def _best_lateral_arm(self, skel: Skeleton):
        if len(skel.landmarks) < 13:
            return None
        w, h = skel.image_width, skel.image_height
        candidates = []
        for side, idx in LATERAL_ARMS.items():
            s, e, hip = skel.landmarks[idx["shoulder"]], skel.landmarks[idx["elbow"]], skel.landmarks[idx["hip"]]
            vis = (s.visibility + e.visibility + hip.visibility) / 3.0
            if vis < self.min_visibility_lateral:
                continue
            sp, ep, hp = _pt(s, w, h), _pt(e, w, h), _pt(hip, w, h)
            abduction = angle_between(hp, sp, ep)   # angle at shoulder between torso-down and arm
            candidates.append({"side": side, "angle": abduction, "confidence": vis,
                                "shoulder": sp, "elbow": ep, "hip": hp})
        return max(candidates, key=lambda a: a["confidence"]) if candidates else None

    def _analyze_lateral_raise(self, skel: Skeleton) -> FormStatus:
        out = FormStatus()
        out.header = skel.header
        arm = self._best_lateral_arm(skel)
        if arm is None:
            out.status = "arm_not_visible"
            out.detail = "shoulder/elbow/hip below visibility threshold"
            out.elbow_angle = float("nan")
            out.confidence = 0.0
            return out

        angle = arm["angle"]
        if angle <= self.lat_down_angle:
            status, detail = "arm_at_side",  "ready position"
        elif angle >= self.lat_top_angle:
            status, detail = "at_top",        "arm at shoulder height"
        elif angle >= self.lat_partial_angle:
            status, detail = "near_top",      "almost at shoulder height"
        else:
            status, detail = "mid_range",     "raising"

        out.status      = status
        out.detail      = f"{arm['side']}: {detail}"
        out.elbow_angle = float(angle)   # shoulder abduction angle
        out.confidence  = float(arm["confidence"])
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
