#!/usr/bin/env python3
"""Display node — home screen (pick exercise + reps) and live workout view.

Two screens:
  HOME     — click an exercise card, set target reps with -/+, click START
  WORKOUT  — live camera, skeleton, rep counter, coach panel

Workout-screen keys:
  1 — switch to Bicep Curl
  2 — switch to Lateral Raise
  3 — switch to Squat
  D — toggle debug overlay
  Q — return to home screen
  Esc — quit the app
"""

import math
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from gymbuddy_ros.msg import FormStatus, IntentUpdate, Skeleton, WorkoutStats

# ── COCO 17-keypoint indices (matches MediaPipe→COCO remap in pose_detection) ── #
LANDMARKS = {
    "nose":           0,
    "l_shoulder":     5, "r_shoulder":     6,
    "l_elbow":        7, "r_elbow":        8,
    "l_wrist":        9, "r_wrist":       10,
    "l_hip":         11, "r_hip":         12,
    "l_knee":        13, "r_knee":        14,
}

# Full skeleton edges drawn in grey
SKELETON_EDGES = [
    (5,  6),
    (5,  7), (7,  9),
    (6,  8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (12, 14),
    (13, 15), (14, 16),   # knee → ankle
]

CURL_ACTIVE_EDGES    = {(5, 7), (7, 9), (6, 8), (8, 10)}
LATERAL_ACTIVE_EDGES = {(5, 11), (5, 7), (6, 12), (6, 8)}
SQUAT_ACTIVE_EDGES   = {(11, 13), (13, 15), (12, 14), (14, 16)}

CANVAS_W, CANVAS_H = 960, 540

# ── Colours ──────────────────────────────────────────────────────────── #
CLR_SKELETON  = (80,  80,  80)
CLR_ACTIVE    = (0,  220, 255)
CLR_JOINT     = (0,  255, 120)
CLR_ARC       = (0,  200, 255)
CLR_PANEL_TXT = (245, 245, 245)
CLR_ACCENT    = (160, 210, 255)
CLR_BG        = (18,  22,  28)
CLR_CARD      = (32,  38,  46)
CLR_CARD_SEL  = (60, 110, 160)
CLR_BTN       = (45,  55,  68)
CLR_BTN_HOT   = (70, 130, 200)
CLR_START_BTN = (40, 170,  90)
CLR_START_HOT = (60, 210, 110)

# Image filenames bundled with the package
EXERCISES = [
    {
        "id":       "bicep_curl",
        "label":    "BICEP CURL",
        "subtitle": "Dumbbell, one arm",
        "image":    "dumbbell-bicep-curl.webp",
    },
    {
        "id":       "lateral_raise",
        "label":    "LATERAL RAISE",
        "subtitle": "Shoulder abduction",
        "image":    "Dumbbell-Lateral-Raise_31c81eee-81c4-4ffe-890d-ee13dd5bbf20_600x600.webp",
    },
    {
        "id":       "squat",
        "label":    "WEIGHTED SQUAT",
        "subtitle": "Goblet / dumbbell hold",
        "image":    "squat.webp",
    },
]

EX_LABELS = {
    "bicep_curl":   "BICEP CURL",
    "lateral_raise": "LATERAL RAISE",
    "squat":        "WEIGHTED SQUAT",
}

SQUAT_CUE_MAP = {
    "squat_standing":   ("STANDING",    (0, 255, 120)),
    "squat_mid":        ("SQUATTING",   (0, 220, 255)),
    "squat_near_depth": ("GOING DEEP",  (0, 220, 255)),
    "squat_at_depth":   ("AT DEPTH",    (0, 255, 120)),
    "squat_lean_fwd":   ("LEAN FWD!",   (0, 120, 255)),
    "squat_knee_cave":  ("KNEES OUT!",  (0, 120, 255)),
}

CLR_ARM_ARC      = (50,  210, 255)   # arm-hold arc: default cyan
CLR_ARM_GOOD     = (0,   220,  80)   # arm in hold range — green
CLR_ARM_BAD      = (0,   100, 255)   # arm dropped / out of range — red-orange
ARM_HOLD_MIN_DIS = 40.0              # display threshold: too compressed below this
ARM_HOLD_MAX_DIS = 120.0             # display threshold: arms dropping above this


def _angle_between(a, b, c):
    """Angle in degrees at vertex b formed by rays b→a and b→c."""
    ba = a - b;  bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _lm_pt(lm, w, h):
    return np.array([lm.x * w, lm.y * h])


def _px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))


def _point_in_rect(pt, rect):
    x, y = pt
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _fit_into(img, target_w, target_h):
    """Resize keeping aspect ratio so img fits inside target_w x target_h."""
    ih, iw = img.shape[:2]
    scale = min(target_w / float(iw), target_h / float(ih))
    new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _draw_angle_arc(frame, center, p1, p2, angle_val, color, radius=48):
    a1 = math.degrees(math.atan2(p1[1] - center[1], p1[0] - center[0]))
    a2 = math.degrees(math.atan2(p2[1] - center[1], p2[0] - center[0]))
    if a1 > a2:
        a1, a2 = a2, a1
    if a2 - a1 > 180:
        a1 += 360
        a1, a2 = a2, a1
    cv2.ellipse(frame, center, (radius, radius), 0, a1, a2, color, 2)
    mid = math.radians((a1 + a2) / 2.0)
    tx = int(center[0] + (radius + 18) * math.cos(mid))
    ty = int(center[1] + (radius + 18) * math.sin(mid))
    cv2.putText(frame, f"{angle_val:.0f}deg", (tx - 20, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


class DisplayNode:
    def __init__(self):
        self._lock = threading.Lock()

        # ── ROS state ─────────────────────────────────────────────────── #
        self._frame:    Optional[np.ndarray] = None
        self._skeleton: Optional[Skeleton]   = None
        self._form:     Optional[FormStatus] = None
        self._stats:    Optional[WorkoutStats] = None
        self._coach_tip:   str   = "Waiting for nodes..."
        self._coach_until: float = 0.0
        self._debug:       bool  = rospy.get_param("~debug", False)

        # ── UI state ──────────────────────────────────────────────────── #
        self._on_home          = True
        self._selected_idx     = 0
        self._target_reps      = int(rospy.get_param("~initial_target", 10))
        self._mouse_pos        = (0, 0)
        self._mouse_clicked    = False    # set by mouse callback, cleared in loop

        # rectangles populated each frame so the mouse handler can hit-test
        self._rect_cards    = []                  # list of (rect, exercise_idx)
        self._rect_minus    = (0, 0, 0, 0)
        self._rect_plus     = (0, 0, 0, 0)
        self._rect_start    = (0, 0, 0, 0)

        # ── exercise images (loaded once) ─────────────────────────────── #
        self._exercise_imgs = self._load_images()

        # ── ROS publishers / subscribers ──────────────────────────────── #
        self._exercise_pub = rospy.Publisher("/active_exercise", String,
                                             queue_size=1, latch=True)
        self._intent_pub   = rospy.Publisher("/intent_update",   IntentUpdate,
                                             queue_size=4)
        # announce default exercise so form_analysis starts in a known mode
        self._exercise_pub.publish(String(data=EXERCISES[self._selected_idx]["id"]))

        rospy.Subscriber("/raw_camera_frame", CompressedImage, self._cb_frame, queue_size=1)
        rospy.Subscriber("/skeleton_data",    Skeleton,        self._cb_skel,  queue_size=1)
        rospy.Subscriber("/form_status",      FormStatus,      self._cb_form,  queue_size=2)
        rospy.Subscriber("/workout_stats",    WorkoutStats,    self._cb_stats, queue_size=2)
        rospy.Subscriber("/coaching_output",  String,          self._cb_coach, queue_size=4)
        rospy.Subscriber("/tts_priority",     String,          self._cb_coach, queue_size=4)

        rospy.loginfo("display_node ready — home screen active")

    # ── Asset loading ─────────────────────────────────────────────────── #

    def _load_images(self):
        try:
            pkg_path = rospkg.RosPack().get_path("gymbuddy_ros")
        except rospkg.ResourceNotFound:
            pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_dir = os.path.join(pkg_path, "images")
        out = []
        for ex in EXERCISES:
            path = os.path.join(img_dir, ex["image"])
            img = cv2.imread(path)
            if img is None:
                rospy.logwarn("display: missing image %s — using placeholder", path)
                img = np.full((400, 400, 3), 60, dtype=np.uint8)
                cv2.putText(img, ex["label"], (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            out.append(img)
        return out

    # ── ROS callbacks ─────────────────────────────────────────────────── #

    def _cb_frame(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            with self._lock:
                self._frame = bgr

    def _cb_skel(self, msg: Skeleton):
        with self._lock:
            self._skeleton = msg

    def _cb_form(self, msg: FormStatus):
        with self._lock:
            self._form = msg

    def _cb_stats(self, msg: WorkoutStats):
        if msg.target_reps != 0:
            with self._lock:
                self._stats = msg

    def _cb_coach(self, msg: String):
        with self._lock:
            self._coach_tip   = msg.data
            self._coach_until = time.time() + 6.0

    # ── Mouse handler ─────────────────────────────────────────────────── #

    def _on_mouse(self, event, x, y, flags, param):
        self._mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_clicked = True

    # ── Home screen ───────────────────────────────────────────────────── #

    def _draw_home(self, frame):
        fh, fw = frame.shape[:2]
        frame[:] = CLR_BG

        # title
        cv2.putText(frame, "GYMBUDDY", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, "Pick an exercise and your target reps.",
                    (42, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 200), 1)

        # exercise cards — dynamically sized so all fit within the canvas
        n      = len(EXERCISES)
        gap    = 20
        card_h = 280
        card_w = min(360, (fw - gap * (n + 1)) // n)
        total_w = card_w * n + gap * (n - 1)
        start_x = (fw - total_w) // 2
        card_y = 120

        self._rect_cards = []
        for i, ex in enumerate(EXERCISES):
            x = start_x + i * (card_w + gap)
            rect = (x, card_y, card_w, card_h)
            selected = (i == self._selected_idx)
            hovered  = _point_in_rect(self._mouse_pos, rect)

            border_clr = CLR_CARD_SEL if selected else (CLR_BTN_HOT if hovered else CLR_CARD)
            fill_clr   = CLR_CARD
            cv2.rectangle(frame, (x, card_y), (x + card_w, card_y + card_h),
                          fill_clr, -1)
            cv2.rectangle(frame, (x, card_y), (x + card_w, card_y + card_h),
                          border_clr, 4 if selected else 2)

            # image area inside card
            img_area_h = card_h - 80
            img = _fit_into(self._exercise_imgs[i], card_w - 24, img_area_h - 12)
            ih, iw = img.shape[:2]
            ix = x + (card_w - iw) // 2
            iy = card_y + 12 + (img_area_h - ih) // 2
            frame[iy:iy + ih, ix:ix + iw] = img

            # label
            cv2.putText(frame, ex["label"], (x + 18, card_y + card_h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(frame, ex["subtitle"], (x + 18, card_y + card_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 195), 1)

            self._rect_cards.append((rect, i))

        # rep selector
        sel_y = card_y + card_h + 40
        cv2.putText(frame, "TARGET REPS", (fw // 2 - 70, sel_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 200), 1)

        btn_w, btn_h = 56, 56
        rep_w = 110
        row_w = btn_w * 2 + rep_w + 24
        row_x = (fw - row_w) // 2
        row_y = sel_y

        minus_rect = (row_x, row_y, btn_w, btn_h)
        plus_rect  = (row_x + btn_w + rep_w + 24, row_y, btn_w, btn_h)
        self._rect_minus = minus_rect
        self._rect_plus  = plus_rect

        for rect, label in [(minus_rect, "-"), (plus_rect, "+")]:
            mx, my, mw, mh = rect
            hot = _point_in_rect(self._mouse_pos, rect)
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh),
                          CLR_BTN_HOT if hot else CLR_BTN, -1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(frame, label,
                        (mx + (mw - tw) // 2, my + mh - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # rep number display between -/+
        num_x = row_x + btn_w + 12
        cv2.rectangle(frame, (num_x, row_y), (num_x + rep_w, row_y + btn_h),
                      (24, 28, 36), -1)
        rep_str = str(self._target_reps)
        (tw, th), _ = cv2.getTextSize(rep_str, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame, rep_str,
                    (num_x + (rep_w - tw) // 2, row_y + (btn_h + th) // 2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        # start button
        sb_w, sb_h = 280, 64
        sb_x = (fw - sb_w) // 2
        sb_y = row_y + btn_h + 30
        start_rect = (sb_x, sb_y, sb_w, sb_h)
        self._rect_start = start_rect
        hot = _point_in_rect(self._mouse_pos, start_rect)
        cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h),
                      CLR_START_HOT if hot else CLR_START_BTN, -1)
        text = "START SET"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        cv2.putText(frame, text,
                    (sb_x + (sb_w - tw) // 2, sb_y + (sb_h + th) // 2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (15, 25, 15), 2)

        cv2.putText(frame, "Esc quits.   In workout: Q returns here.",
                    (40, fh - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (140, 150, 160), 1)

    def _handle_home_click(self):
        # cards
        for rect, idx in self._rect_cards:
            if _point_in_rect(self._mouse_pos, rect):
                if self._selected_idx != idx:
                    self._selected_idx = idx
                    self._exercise_pub.publish(
                        String(data=EXERCISES[idx]["id"]))
                return
        # minus
        if _point_in_rect(self._mouse_pos, self._rect_minus):
            self._target_reps = max(1, self._target_reps - 1)
            return
        # plus
        if _point_in_rect(self._mouse_pos, self._rect_plus):
            self._target_reps = min(99, self._target_reps + 1)
            return
        # start
        if _point_in_rect(self._mouse_pos, self._rect_start):
            self._start_set()

    def _start_set(self):
        ex_id = EXERCISES[self._selected_idx]["id"]
        # ensure form_analysis & rep_counter are on the right exercise
        self._exercise_pub.publish(String(data=ex_id))

        # set target on workout_manager
        upd = IntentUpdate()
        upd.header.stamp = rospy.Time.now()
        upd.action = "update_target"
        upd.value  = int(self._target_reps)
        upd.text   = "from display home"
        self._intent_pub.publish(upd)

        # start the set
        start = IntentUpdate()
        start.header.stamp = rospy.Time.now()
        start.action = "start_set"
        start.value  = 0
        start.text   = ""
        self._intent_pub.publish(start)

        # reset stats so the workout view begins clean
        with self._lock:
            self._stats = None
            self._form  = None
            self._on_home = False
        rospy.loginfo("display: started %s, target=%d", ex_id, self._target_reps)

    def _return_to_home(self):
        # stop the active set
        stop = IntentUpdate()
        stop.header.stamp = rospy.Time.now()
        stop.action = "stop_set"
        stop.value  = 0
        stop.text   = "user returned to home"
        self._intent_pub.publish(stop)
        with self._lock:
            self._on_home = True
        rospy.loginfo("display: returned to home")

    # ── Workout: skeleton drawing ─────────────────────────────────────── #

    def _draw_skeleton(self, frame, skeleton: Skeleton, form: Optional[FormStatus],
                       exercise: str):
        if skeleton is None or len(skeleton.landmarks) < 17:
            return
        w, h = skeleton.image_width, skeleton.image_height
        lms = skeleton.landmarks

        if exercise == "bicep_curl":
            active_edges = CURL_ACTIVE_EDGES
        elif exercise == "squat":
            active_edges = SQUAT_ACTIVE_EDGES
        else:
            active_edges = LATERAL_ACTIVE_EDGES

        for i, j in SKELETON_EDGES:
            if i >= len(lms) or j >= len(lms):
                continue
            vis = (lms[i].visibility + lms[j].visibility) / 2.0
            if vis < 0.1:
                continue
            p1, p2 = _px(lms[i], w, h), _px(lms[j], w, h)
            is_active = (i, j) in active_edges or (j, i) in active_edges
            color = CLR_ACTIVE if is_active else CLR_SKELETON
            thickness = 3 if is_active else 1
            alpha = min(1.0, vis * 1.5)
            blended = tuple(int(c * alpha) for c in color)
            cv2.line(frame, p1, p2, blended, thickness)

        for idx in range(min(17, len(lms))):
            lm = lms[idx]
            if lm.visibility < 0.2:
                continue
            px = _px(lm, w, h)
            large = (5, 6, 7, 8, 9, 10, 11, 12)
            if exercise == "squat":
                large = large + (13, 14, 15, 16)
            radius = 6 if idx in large else 4
            cv2.circle(frame, px, radius, CLR_JOINT, -1)

        if form is not None:
            if exercise == "squat":
                # Knee arcs — angle from FormStatus (hip→knee→ankle)
                squat_map = {
                    "left":  {"hip": 11, "knee": 13, "ankle": 15, "angle": form.elbow_angle},
                    "right": {"hip": 12, "knee": 14, "ankle": 16, "angle": form.right_elbow_angle},
                }
                for side, idx_map in squat_map.items():
                    angle_val = idx_map["angle"]
                    if angle_val != angle_val:
                        continue
                    if idx_map["ankle"] >= len(lms):
                        continue
                    hip_lm  = lms[idx_map["hip"]]
                    knee_lm = lms[idx_map["knee"]]
                    ank_lm  = lms[idx_map["ankle"]]
                    if knee_lm.visibility < 0.3 or hip_lm.visibility < 0.3:
                        continue
                    _draw_angle_arc(frame,
                                    _px(knee_lm, w, h),
                                    _px(hip_lm, w, h),
                                    _px(ank_lm, w, h),
                                    angle_val, CLR_ARC)

                # Arm-hold arcs — use FormStatus.aux_angle (pre-computed by form_analysis)
                # Green = good hold (40-120°), Red = arms dropping (>120°)
                arm_map = {
                    "left":  (5, 7, 9,
                               form.aux_angle if form is not None else float("nan")),
                    "right": (6, 8, 10,
                               form.right_aux_angle if form is not None else float("nan")),
                }
                for side, (si, ei, wi, precomp_angle) in arm_map.items():
                    if wi >= len(lms):
                        continue
                    s_lm, e_lm, wt_lm = lms[si], lms[ei], lms[wi]
                    if e_lm.visibility < 0.15 or s_lm.visibility < 0.15:
                        continue
                    # Prefer pre-computed angle; fall back to live if NaN
                    if precomp_angle == precomp_angle:
                        arm_angle = precomp_angle
                    else:
                        arm_angle = _angle_between(
                            _lm_pt(s_lm, w, h), _lm_pt(e_lm, w, h), _lm_pt(wt_lm, w, h))
                    if arm_angle != arm_angle:
                        continue
                    if ARM_HOLD_MIN_DIS <= arm_angle <= ARM_HOLD_MAX_DIS:
                        arc_color = CLR_ARM_GOOD
                    else:
                        arc_color = CLR_ARM_BAD
                    _draw_angle_arc(frame,
                                    _px(e_lm, w, h),
                                    _px(s_lm, w, h),
                                    _px(wt_lm, w, h),
                                    arm_angle, arc_color, radius=36)
            else:
                side_map = {
                    "left":  {"s": 5, "e": 7, "w":  9, "h": 11, "angle": form.elbow_angle},
                    "right": {"s": 6, "e": 8, "w": 10, "h": 12, "angle": form.right_elbow_angle},
                }
                for side, idx_map in side_map.items():
                    angle_val = idx_map["angle"]
                    if angle_val != angle_val:   # NaN
                        continue
                    s_lm = lms[idx_map["s"]]
                    e_lm = lms[idx_map["e"]]
                    w_lm = lms[idx_map["w"]]
                    h_lm = lms[idx_map["h"]]
                    if s_lm.visibility < 0.3 or e_lm.visibility < 0.3:
                        continue
                    if exercise == "bicep_curl":
                        center = _px(e_lm, w, h)
                        p1     = _px(s_lm, w, h)
                        p2     = _px(w_lm, w, h)
                    else:
                        center = _px(s_lm, w, h)
                        p1     = _px(h_lm, w, h)
                        p2     = _px(e_lm, w, h)
                    _draw_angle_arc(frame, center, p1, p2, angle_val, CLR_ARC)

    # ── Workout: panel ────────────────────────────────────────────────── #

    @staticmethod
    def _wrap(frame, text, x, y, max_w, color, scale=0.52, thick=1, gap=22):
        line = ""
        for word in text.split():
            candidate = word if not line else f"{line} {word}"
            (lw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            if line and lw > max_w:
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
                y += gap
                line = word
            else:
                line = candidate
        if line:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def _draw_panel(self, frame, form, stats, tip, coach_active, debug, exercise, skeleton=None):
        fh, fw = frame.shape[:2]
        pw = 390
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (pw, fh), CLR_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        ex_label = EX_LABELS.get(exercise, exercise.upper().replace("_", " "))
        cv2.putText(frame, "GYMBUDDY", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, ex_label,   (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_ACCENT, 1)

        status_lbl = "LIVE SET" if (stats and stats.target_reps > 0) else "READY"
        cv2.putText(frame, status_lbl, (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (190, 190, 190), 2)

        clean  = stats.clean_reps     if stats else 0
        total  = stats.total_attempts if stats else 0
        target = stats.target_reps    if stats else self._target_reps
        rep_str = f"{str(clean).zfill(2)}/{target}" if target > 0 else str(clean).zfill(2)
        rep_col = (0, 255, 120) if (target > 0 and clean >= target) else (255, 255, 255)
        cv2.putText(frame, rep_str, (30, 222), cv2.FONT_HERSHEY_SIMPLEX, 3.2, rep_col, 7)
        cv2.putText(frame, "CLEAN REPS",      (36, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (190, 190, 190), 2)
        cv2.putText(frame, f"ATTEMPTS {total}", (38, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (190, 190, 190), 2)

        CUE_MAP = {
            "fully_extended": ("READY",        (0, 255, 120)),
            "mid_range":      ("CURLING",       (0, 220, 255)),
            "near_top":       ("ALMOST THERE",  (0, 220, 255)),
            "depth_reached":  ("TOP",           (0, 255, 120)),
            "arm_at_side":    ("READY",         (0, 255, 120)),
            "at_top":         ("TOP",           (0, 255, 120)),
            **SQUAT_CUE_MAP,
        }
        if form is not None:
            cue_txt, cue_col = CUE_MAP.get(form.status,
                (form.status.replace("_", " ").upper(), (0, 220, 255)))
            if form.status == "arm_not_visible":
                cue_txt, cue_col = "NOT VISIBLE", (100, 100, 100)
        else:
            cue_txt, cue_col = "NO SKELETON", (80, 80, 80)
        cv2.putText(frame, cue_txt, (30, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.72, cue_col, 2)

        if form is not None:
            if exercise == "squat":
                # Show knee angles (primary) + live arm-hold angles from skeleton
                l_k = f"{form.elbow_angle:.0f}" if form.elbow_angle == form.elbow_angle else "--"
                r_k = f"{form.right_elbow_angle:.0f}" if form.right_elbow_angle == form.right_elbow_angle else "--"
                cv2.putText(frame, f"KNEE  L:{l_k}deg  R:{r_k}deg",
                            (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.46, CLR_ACCENT, 1)

                # Depth percentage — how close to SQUAT_DEPTH (105°) from standing (150°)
                best_k = None
                if form.elbow_angle == form.elbow_angle:
                    best_k = form.elbow_angle
                if form.right_elbow_angle == form.right_elbow_angle:
                    v = form.right_elbow_angle
                    best_k = v if best_k is None else min(best_k, v)
                if best_k is not None:
                    depth_pct = max(0, min(100, int((150 - best_k) / (150 - 105) * 100)))
                    col = (0, 200, 100) if depth_pct >= 100 else ((0, 190, 255) if depth_pct >= 60 else (0, 130, 255))
                    cv2.putText(frame, f"DEPTH {depth_pct}%",
                                (30, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

                # Arm-hold angles from FormStatus — color-coded for hold quality
                l_arm_a = form.aux_angle
                r_arm_a = form.right_aux_angle
                l_arm_str = f"{l_arm_a:.0f}" if l_arm_a == l_arm_a else "--"
                r_arm_str = f"{r_arm_a:.0f}" if r_arm_a == r_arm_a else "--"
                l_bad = (l_arm_a == l_arm_a) and not (ARM_HOLD_MIN_DIS <= l_arm_a <= ARM_HOLD_MAX_DIS)
                r_bad = (r_arm_a == r_arm_a) and not (ARM_HOLD_MIN_DIS <= r_arm_a <= ARM_HOLD_MAX_DIS)
                arm_col = CLR_ARM_BAD if (l_bad or r_bad) else CLR_ARM_GOOD
                cv2.putText(frame, f"ARM   L:{l_arm_str}deg  R:{r_arm_str}deg",
                            (30, 388), cv2.FONT_HERSHEY_SIMPLEX, 0.46, arm_col, 1)
            else:
                angle_label = "elbow" if exercise == "bicep_curl" else "abduction"
                l_str = f"{form.elbow_angle:.1f}" if form.elbow_angle == form.elbow_angle else "--"
                r_str = f"{form.right_elbow_angle:.1f}" if form.right_elbow_angle == form.right_elbow_angle else "--"
                cv2.putText(frame, f"L {angle_label}: {l_str}deg  R: {r_str}deg",
                            (30, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.46, CLR_ACCENT, 1)

        coach_y = 410 if exercise == "squat" else 378
        cv2.putText(frame, "AI COACH", (30, coach_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_ACCENT, 1)
        self._wrap(frame, tip if coach_active else "Waiting...",
                   30, coach_y + 24, pw - 50, CLR_PANEL_TXT)

        if stats and stats.last_rep_issue:
            issue_txt = stats.last_rep_issue.replace(",", "  ").replace("_", " ")
            cv2.putText(frame, f"last: {issue_txt}", (30, fh - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 130, 130), 1)

        cv2.putText(frame, "1:Curl  2:Raise  3:Squat  D:debug  Q:home  Esc:quit",
                    (30, fh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

        if debug and form is not None:
            l_str = f"{form.elbow_angle:.1f}" if form.elbow_angle == form.elbow_angle else "--"
            r_str = f"{form.right_elbow_angle:.1f}" if form.right_elbow_angle == form.right_elbow_angle else "--"
            lines = [
                f"exercise: {exercise}",
                f"status: {form.status}",
                f"detail: {form.detail}",
                f"L angle: {l_str}  R angle: {r_str}",
                f"conf L:{form.confidence:.2f} R:{form.right_confidence:.2f}",
                f"reps: {clean}/{total}",
                f"issue: {stats.last_rep_issue or 'none'}" if stats else "",
            ]
            for i, txt in enumerate(l for l in lines if l):
                cv2.putText(frame, txt, (fw - 330, 28 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1)

    # ── Main loop ─────────────────────────────────────────────────────── #

    def spin(self):
        cv2.namedWindow("GymBuddy ROS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GymBuddy ROS", CANVAS_W, CANVAS_H)
        cv2.setMouseCallback("GymBuddy ROS", self._on_mouse)

        while not rospy.is_shutdown():
            with self._lock:
                on_home   = self._on_home
                raw       = self._frame
                skeleton  = self._skeleton
                form      = self._form
                stats     = self._stats
                tip       = self._coach_tip
                coach_on  = time.time() < self._coach_until
                debug     = self._debug
                exercise  = EXERCISES[self._selected_idx]["id"]
                clicked   = self._mouse_clicked
                self._mouse_clicked = False

            if on_home:
                canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                self._draw_home(canvas)
                if clicked:
                    self._handle_home_click()
                cv2.imshow("GymBuddy ROS", canvas)
            else:
                if raw is not None:
                    frame = raw.copy()
                else:
                    frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for camera...",
                                (CANVAS_W // 2 - 170, CANVAS_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
                self._draw_skeleton(frame, skeleton, form, exercise)
                self._draw_panel(frame, form, stats, tip, coach_on, debug, exercise, skeleton)
                cv2.imshow("GymBuddy ROS", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:                              # Esc → quit
                rospy.signal_shutdown("user quit")
                break
            if key == ord("q"):
                if on_home:
                    rospy.signal_shutdown("user quit")
                    break
                self._return_to_home()
                continue
            if not on_home:
                if key == ord("d"):
                    with self._lock:
                        self._debug = not self._debug
                elif key == ord("1"):
                    with self._lock:
                        self._selected_idx = 0
                    self._exercise_pub.publish(String(data="bicep_curl"))
                    rospy.loginfo("display: switched to bicep_curl")
                elif key == ord("2"):
                    with self._lock:
                        self._selected_idx = 1
                    self._exercise_pub.publish(String(data="lateral_raise"))
                    rospy.loginfo("display: switched to lateral_raise")
                elif key == ord("3"):
                    with self._lock:
                        self._selected_idx = 2
                    self._exercise_pub.publish(String(data="squat"))
                    rospy.loginfo("display: switched to squat")

        cv2.destroyAllWindows()


def main():
    rospy.init_node("display_node")
    node = DisplayNode()
    node.spin()


if __name__ == "__main__":
    main()
