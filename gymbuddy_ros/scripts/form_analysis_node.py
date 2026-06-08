#!/usr/bin/env python3
"""Form analysis for bicep curl, lateral raise, and squat.

Publishes FormStatus on every skeleton frame.
  elbow_angle field carries:
    bicep_curl    — elbow flexion angle  (155° = arm straight, 90° = full curl)
    lateral_raise — shoulder abduction angle (0° = arm at side, 90° = horizontal)
    squat         — knee flexion angle   (165° = standing, 90° = parallel squat)
"""

import math
import threading
import time

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
SQUAT_LEGS = {
    "left":  {"hip": 11, "knee": 13, "ankle": 15},
    "right": {"hip": 12, "knee": 14, "ankle": 16},
}

# Bicep curl thresholds — must stay in sync with rep_counter_node.py
CURL_EXTENDED    = 145.0
CURL_TOP         = 95.0
CURL_PARTIAL     = 115.0

# Lateral raise thresholds — shoulder abduction angle
LAT_DOWN         = 30.0   # arm at side
LAT_PARTIAL      = 55.0   # partial raise
LAT_TOP          = 75.0   # arm at shoulder height

# Weighted squat thresholds — knee flexion angle (hip→knee→ankle), lenient
SQUAT_STANDING   = 150.0  # upright / standing position
SQUAT_PARTIAL    = 120.0  # partial depth (just below parallel)
SQUAT_DEPTH      = 105.0  # target depth — achievable quarter-to-half squat

SQUAT_LEAN_THRESH  = 42.0   # torso degrees from vertical before flagging forward lean
SQUAT_VALGUS_RATIO = 0.65   # knee gap / ankle gap below this = knee cave
ARM_HOLD_MAX       = 120.0  # elbow angle above this = arms dropping / weight not held
ARM_HOLD_MIN       = 40.0   # elbow angle below this = arms too compressed
ARM_VIS_MIN        = 0.10   # min arm landmark visibility for arm angle check
COACHING_INTERVAL  = 8.0    # seconds between the same form coaching tip

# Bicep curl only needs shoulder+elbow+wrist visible — keep the bar low so
# close-up framing (where hips often leave the frame) still works.
MIN_VISIBILITY_CURL    = 0.20
# Lateral raise requires the hip as the torso reference, so we keep this higher.
MIN_VISIBILITY_LATERAL = 0.35
MIN_VISIBILITY_SQUAT   = 0.04  # low: legs often partially out of frame


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
        self.min_visibility_squat   = float(rospy.get_param("~min_visibility_squat",   MIN_VISIBILITY_SQUAT))

        self._exercise = "bicep_curl"
        self._lock = threading.Lock()
        self._last_coach: dict = {}   # key → timestamp of last coaching message

        self.pub          = rospy.Publisher("/form_status",     FormStatus, queue_size=4)
        self.coaching_pub = rospy.Publisher("/coaching_output", String,     queue_size=2)
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
    # Squat — helpers                                                      #
    # ------------------------------------------------------------------ #

    def _maybe_coach(self, key: str, text: str):
        """Publish a coaching tip at most once per COACHING_INTERVAL seconds per key."""
        now = time.time()
        if now - self._last_coach.get(key, 0) >= COACHING_INTERVAL:
            self.coaching_pub.publish(String(data=text))
            self._last_coach[key] = now

    def _torso_lean_angle(self, skel: Skeleton) -> float:
        """Torso lean from vertical in degrees (0=upright, 90=horizontal)."""
        if len(skel.landmarks) < 13:
            return float("nan")
        w, h = skel.image_width, skel.image_height
        l_sh, r_sh   = skel.landmarks[5],  skel.landmarks[6]
        l_hip, r_hip = skel.landmarks[11], skel.landmarks[12]
        if min(l_sh.visibility, r_sh.visibility,
               l_hip.visibility, r_hip.visibility) < 0.20:
            return float("nan")
        sh_x  = (l_sh.x  + r_sh.x)  / 2 * w
        sh_y  = (l_sh.y  + r_sh.y)  / 2 * h
        hip_x = (l_hip.x + r_hip.x) / 2 * w
        hip_y = (l_hip.y + r_hip.y) / 2 * h
        dy = hip_y - sh_y   # positive = hip below shoulder (upright)
        if dy <= 0:
            return 90.0
        return math.degrees(math.atan2(abs(sh_x - hip_x), dy))

    def _knee_valgus(self, skel: Skeleton) -> bool:
        """True if knees are collapsing inward relative to ankle width."""
        if len(skel.landmarks) < 17:
            return False
        lm = skel.landmarks
        if min(lm[13].visibility, lm[14].visibility,
               lm[15].visibility, lm[16].visibility) < 0.25:
            return False
        w = skel.image_width
        knee_gap  = abs(lm[13].x - lm[14].x) * w
        ankle_gap = abs(lm[15].x - lm[16].x) * w
        if ankle_gap < 8:
            return False
        return knee_gap < ankle_gap * SQUAT_VALGUS_RATIO

    def _get_arm_angle(self, skel: Skeleton, side: str):
        """Elbow flexion angle for weighted squat arm hold (shoulder→elbow→wrist)."""
        if len(skel.landmarks) < 11:
            return None
        w, h = skel.image_width, skel.image_height
        idx = CURL_ARMS[side]
        s  = skel.landmarks[idx["shoulder"]]
        e  = skel.landmarks[idx["elbow"]]
        wt = skel.landmarks[idx["wrist"]]
        if min(s.visibility, e.visibility, wt.visibility) < ARM_VIS_MIN:
            return None
        return angle_between(_pt(s, w, h), _pt(e, w, h), _pt(wt, w, h))

    def _get_squat_leg(self, skel: Skeleton, side: str):
        if len(skel.landmarks) < 17:
            return None
        w, h = skel.image_width, skel.image_height
        idx = SQUAT_LEGS[side]
        hip  = skel.landmarks[idx["hip"]]
        knee = skel.landmarks[idx["knee"]]
        ank  = skel.landmarks[idx["ankle"]]
        # Require only knee to be visible — ankle often leaves frame
        if knee.visibility < self.min_visibility_squat:
            return None
        conf = (hip.visibility + knee.visibility + ank.visibility) / 3.0
        angle = angle_between(_pt(hip, w, h), _pt(knee, w, h), _pt(ank, w, h))
        return {"side": side, "angle": angle, "confidence": conf}

    def _squat_status(self, angle: float) -> tuple:
        if angle >= SQUAT_STANDING:
            return "squat_standing",    "standing"
        elif angle > SQUAT_PARTIAL:
            return "squat_mid",         "going down"
        elif angle > SQUAT_DEPTH:
            return "squat_near_depth",  "almost at depth"
        else:
            return "squat_at_depth",    "at depth"

    def _analyze_squat(self, skel: Skeleton) -> FormStatus:
        out = FormStatus()
        out.header = skel.header

        left  = self._get_squat_leg(skel, "left")
        right = self._get_squat_leg(skel, "right")

        if left is None and right is None:
            out.status = "arm_not_visible"
            out.detail = "legs below visibility threshold"
            out.elbow_angle       = float("nan")
            out.confidence        = 0.0
            out.right_elbow_angle = float("nan")
            out.right_confidence  = 0.0
            return out

        if left is not None:
            l_status, l_detail = self._squat_status(left["angle"])
            out.elbow_angle = float(left["angle"])
            out.confidence  = float(left["confidence"])
        else:
            l_status = "arm_not_visible"
            l_detail = "not visible"
            out.elbow_angle = float("nan")
            out.confidence  = 0.0

        if right is not None:
            r_status, r_detail = self._squat_status(right["angle"])
            out.right_elbow_angle = float(right["angle"])
            out.right_confidence  = float(right["confidence"])
        else:
            r_status = "arm_not_visible"
            r_detail = "not visible"
            out.right_elbow_angle = float("nan")
            out.right_confidence  = 0.0

        # worst = leg furthest from depth (highest angle = least squatted)
        if left is not None and right is not None:
            worst = left if left["angle"] > right["angle"] else right
            out.status = self._squat_status(worst["angle"])[0]
            best_angle = min(left["angle"], right["angle"])
            out.detail = f"L:{left['angle']:.0f}deg R:{right['angle']:.0f}deg"
        elif left is not None:
            out.status = l_status
            best_angle = left["angle"]
            out.detail = f"left: {l_detail} (right not visible)"
        else:
            out.status = r_status
            best_angle = right["angle"]
            out.detail = f"right: {r_detail} (left not visible)"

        # Arm hold angles for weighted squat — always publish so display can color them
        l_arm = self._get_arm_angle(skel, "left")
        r_arm = self._get_arm_angle(skel, "right")
        out.aux_angle       = float(l_arm) if l_arm is not None else float("nan")
        out.right_aux_angle = float(r_arm) if r_arm is not None else float("nan")

        # Form quality checks — only when actually squatting
        if best_angle < SQUAT_PARTIAL:
            form_flags = []
            lean = self._torso_lean_angle(skel)
            if lean == lean and lean > SQUAT_LEAN_THRESH:
                form_flags.append(f"lean:{lean:.0f}deg")
                self._maybe_coach("lean",
                    "Keep your chest up! You're leaning too far forward.")
                out.status = "squat_lean_fwd"

            if self._knee_valgus(skel):
                form_flags.append("knee_cave")
                self._maybe_coach("valgus",
                    "Drive your knees out! Push them over your toes.")
                if out.status == self._squat_status(best_angle)[0]:
                    out.status = "squat_knee_cave"

            # Arm hold check — goblet / weighted squat position
            l_bad = l_arm is not None and l_arm > ARM_HOLD_MAX
            r_bad = r_arm is not None and r_arm > ARM_HOLD_MAX
            if l_bad or r_bad:
                form_flags.append("arms_drop")
                self._maybe_coach("arm_hold",
                    "Hold weight at chest! Keep elbows bent and arms in tight.")

            if form_flags:
                out.detail += "  |  " + "  ".join(form_flags)


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
        elif exercise == "squat":
            self.pub.publish(self._analyze_squat(msg))
        else:
            self.pub.publish(self._analyze_bicep_curl(msg))


def main():
    rospy.init_node("form_analysis_node")
    FormAnalysisNode()
    rospy.spin()


if __name__ == "__main__":
    main()
