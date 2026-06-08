#!/usr/bin/env python3
"""Exercise-aware rep counter — dual-arm mode.

Both arms must complete the full range of motion for a rep to count.
Each arm is tracked by its own state machine. A rep is registered only
when BOTH arm machines finish in the same rep window.

Bicep curl  — elbow flexion angle; issues: partial_curl, incomplete_extension, too_fast
Lateral raise — shoulder abduction angle; issues: partial_raise, no_return, too_fast
"""

import threading
import time

import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import FormStatus, IntentUpdate, WorkoutStats

# Bicep curl thresholds
CURL_EXTENDED      = 145.0
CURL_CURLING       = 130.0
CURL_TOP           = 95.0
CURL_PARTIAL_TOP   = 115.0

# Lateral raise thresholds
LAT_DOWN           = 30.0
LAT_START          = 32.0
LAT_PARTIAL        = 55.0
LAT_TOP            = 75.0

# Weighted squat thresholds — lenient so a quarter-to-half squat counts
SQUAT_STANDING     = 150.0  # upright position
SQUAT_START        = 145.0  # angle below this triggers a rep start
SQUAT_PARTIAL      = 120.0  # partial depth (passes through here on way down)
SQUAT_DEPTH        = 105.0  # target depth — achievable for most people
SQUAT_FAST_REP     = 0.7    # squats faster than this are flagged too_fast
SQUAT_ARM_HOLD_MAX = 120.0  # elbow angle above this = arms not holding weight
SQUAT_ARM_DROP_RATIO = 0.30 # fraction of in-rep frames with dropped arms = issue

FAST_REP_SECONDS   = 0.6

# How long to wait (seconds) for the second arm to finish after the first.
# If it doesn't finish within this window, the attempt is discarded.
SYNC_WINDOW        = 3.0


class ArmState:
    """Single-arm rep state machine (curl or lateral raise)."""

    def __init__(self, side: str):
        self.side = side
        self.reset_rep()

    def reset_rep(self):
        self.state          = "ready"
        self.rep_started_at = None
        self.best_angle     = None
        self.saw_top        = False
        self.saw_partial    = False
        self.done           = False   # True once this arm finished a rep attempt
        self.issues         = []
        self.duration       = 0.0

    # ------------------------------------------------------------------ #
    # Bicep curl                                                           #
    # ------------------------------------------------------------------ #

    def tick_curl(self, angle: float,
                  extended, curling, top_angle, partial_top, fast_rep):
        if self.done:
            return

        if self.best_angle is None:
            self.best_angle = angle
        self.best_angle = min(self.best_angle, angle)

        if self.state == "ready":
            if angle < curling:
                self.rep_started_at = time.time()
                self.best_angle     = angle
                self.saw_top = self.saw_partial = False
                self.state = "curling"

        elif self.state == "curling":
            if angle <= top_angle:
                self.saw_top = True
                self.state = "top"
            elif angle <= partial_top:
                self.saw_partial = True
            elif angle >= extended and (self.saw_partial or self.best_angle < curling):
                self._finish_curl(True, extended, fast_rep)

        elif self.state == "top":
            if angle > top_angle + 15:
                self.state = "lowering"

        elif self.state == "lowering":
            if angle >= extended:
                self._finish_curl(True, extended, fast_rep)
            elif angle < curling - 10:
                self._finish_curl(False, extended, fast_rep)

    def _finish_curl(self, full_extension: bool, extended, fast_rep):
        if self.rep_started_at is None:
            self.reset_rep()
            return
        self.duration = time.time() - self.rep_started_at
        self.issues = []
        if not self.saw_top:
            self.issues.append("partial_curl")
        if not full_extension:
            self.issues.append("incomplete_extension")
        if self.duration < fast_rep:
            self.issues.append("too_fast")
        self.done = True

    # ------------------------------------------------------------------ #
    # Lateral raise                                                        #
    # ------------------------------------------------------------------ #

    def tick_lateral(self, angle: float, fast_rep):
        if self.done:
            return

        if self.best_angle is None:
            self.best_angle = angle
        self.best_angle = max(self.best_angle, angle)

        if self.state == "ready":
            if angle > LAT_START:
                self.rep_started_at = time.time()
                self.best_angle     = angle
                self.saw_top = self.saw_partial = False
                self.state = "raising"

        elif self.state == "raising":
            if angle >= LAT_TOP:
                self.saw_top = True
                self.state = "at_top"
            elif angle >= LAT_PARTIAL:
                self.saw_partial = True
            elif angle <= LAT_DOWN - 5:
                self._finish_lateral(True, fast_rep)

        elif self.state == "at_top":
            if angle < LAT_TOP - 10:
                self.state = "lowering"

        elif self.state == "lowering":
            if angle <= LAT_DOWN:
                self._finish_lateral(True, fast_rep)
            elif angle >= LAT_TOP:
                self.state = "at_top"

    def _finish_lateral(self, returned: bool, fast_rep):
        if self.rep_started_at is None:
            self.reset_rep()
            return
        self.duration = time.time() - self.rep_started_at
        self.issues = []
        if not self.saw_top:
            self.issues.append("partial_raise")
        if not returned:
            self.issues.append("no_return")
        if self.duration < fast_rep:
            self.issues.append("too_fast")
        self.done = True


    # ------------------------------------------------------------------ #
    # Squat                                                                #
    # ------------------------------------------------------------------ #

    def tick_squat(self, angle: float):
        if self.done:
            return

        if self.best_angle is None:
            self.best_angle = angle
        self.best_angle = min(self.best_angle, angle)  # track deepest (lowest angle)

        if self.state == "ready":
            if angle < SQUAT_START:
                self.rep_started_at = time.time()
                self.best_angle     = angle
                self.saw_top = self.saw_partial = False
                self.state = "descending"

        elif self.state == "descending":
            if angle <= SQUAT_DEPTH:
                self.saw_top = True      # reuse flag: "saw_top" = reached squat depth
                self.state = "bottom"
            elif angle <= SQUAT_PARTIAL:
                self.saw_partial = True
            elif angle >= SQUAT_STANDING:
                self._finish_squat(True)

        elif self.state == "bottom":
            if angle > SQUAT_DEPTH + 15:
                self.state = "ascending"

        elif self.state == "ascending":
            if angle >= SQUAT_STANDING:
                self._finish_squat(True)
            elif angle <= SQUAT_DEPTH:
                self.state = "bottom"

    def _finish_squat(self, returned: bool):
        if self.rep_started_at is None:
            self.reset_rep()
            return
        self.duration = time.time() - self.rep_started_at
        self.issues = []
        if not self.saw_top:
            self.issues.append("partial_squat")
        if not returned:
            self.issues.append("no_return")
        if self.duration < SQUAT_FAST_REP:
            self.issues.append("too_fast")
        self.done = True



class RepCounterNode:
    def __init__(self):
        self.extended_angle    = float(rospy.get_param("~extended_angle",    CURL_EXTENDED))
        self.curling_angle     = float(rospy.get_param("~curling_angle",     CURL_CURLING))
        self.top_angle         = float(rospy.get_param("~top_angle",         CURL_TOP))
        self.partial_top_angle = float(rospy.get_param("~partial_top_angle", CURL_PARTIAL_TOP))
        self.fast_rep_seconds  = float(rospy.get_param("~fast_rep_seconds",  FAST_REP_SECONDS))

        self._exercise = "bicep_curl"
        self._lock = threading.Lock()

        self.pub = rospy.Publisher("/workout_stats", WorkoutStats, queue_size=4, latch=True)
        rospy.Subscriber("/form_status",    FormStatus,   self.on_form,     queue_size=4)
        rospy.Subscriber("/intent_update",  IntentUpdate, self.on_intent,   queue_size=4)
        rospy.Subscriber("/active_exercise", String,      self.on_exercise, queue_size=1)

        self._reset_counters()
        self.active = False
        rospy.loginfo("rep_counter_node ready (dual-arm mode)")

    # ------------------------------------------------------------------ #
    # Exercise switching                                                   #
    # ------------------------------------------------------------------ #

    def on_exercise(self, msg: String):
        with self._lock:
            self._exercise = msg.data
        self._reset_counters()
        self.publish_stats()
        rospy.loginfo("rep_counter: switched to %s", msg.data)

    # ------------------------------------------------------------------ #
    # State                                                                #
    # ------------------------------------------------------------------ #

    def _reset_counters(self):
        self.clean_reps     = 0
        self.total_attempts = 0
        self._left  = ArmState("left")
        self._right = ArmState("right")
        self._rep_start_time = None   # when the first arm began moving
        self._squat_arm_drop_frames = 0
        self._squat_total_frames    = 0

    def _reset_rep(self):
        self._left.reset_rep()
        self._right.reset_rep()
        self._rep_start_time        = None
        self._squat_arm_drop_frames = 0
        self._squat_total_frames    = 0

    def publish_stats(self, last_seconds=0.0, best_angle=None, issue=""):
        with self._lock:
            exercise = self._exercise
        if best_angle is None:
            best_angle = 180.0 if exercise in ("bicep_curl", "squat") else 0.0
        msg = WorkoutStats()
        msg.header.stamp     = rospy.Time.now()
        msg.exercise         = exercise
        msg.clean_reps       = self.clean_reps
        msg.total_attempts   = self.total_attempts
        msg.target_reps      = 0
        msg.last_rep_seconds = float(last_seconds)
        msg.last_min_angle   = float(best_angle)
        msg.last_rep_issue   = issue
        self.pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Intent                                                               #
    # ------------------------------------------------------------------ #

    def on_intent(self, msg: IntentUpdate):
        if msg.action == "start_set":
            self.active = True
            self._reset_counters()
            self.publish_stats()
            rospy.loginfo("rep_counter: set started")
        elif msg.action == "stop_set":
            self.active = False
            self._reset_rep()
            rospy.loginfo("rep_counter: set stopped")
        elif msg.action == "reset":
            self.active = False
            self._reset_counters()
            self.publish_stats()
            rospy.loginfo("rep_counter: reset")

    # ------------------------------------------------------------------ #
    # Form callback                                                        #
    # ------------------------------------------------------------------ #

    def on_form(self, msg: FormStatus):
        if not self.active:
            return

        l_angle = msg.elbow_angle
        r_angle = msg.right_elbow_angle

        with self._lock:
            exercise = self._exercise

        if exercise == "lateral_raise":
            self._tick_lateral(l_angle, r_angle)
        elif exercise == "squat":
            self._track_squat_arm_hold(msg.aux_angle, msg.right_aux_angle)
            self._tick_squat(l_angle, r_angle)
        else:
            self._tick_curl(l_angle, r_angle)

        self._check_sync()

    # ------------------------------------------------------------------ #
    # Dual-arm curl                                                        #
    # ------------------------------------------------------------------ #

    def _tick_curl(self, l_angle: float, r_angle: float):
        # Track when either arm starts moving (for sync timeout)
        for arm, angle in ((self._left, l_angle), (self._right, r_angle)):
            if angle != angle:   # NaN — arm not visible
                continue
            if arm.state == "ready" and angle < self.curling_angle:
                if self._rep_start_time is None:
                    self._rep_start_time = time.time()
            arm.tick_curl(angle,
                          self.extended_angle, self.curling_angle,
                          self.top_angle, self.partial_top_angle,
                          self.fast_rep_seconds)

    # ------------------------------------------------------------------ #
    # Dual-arm lateral raise                                               #
    # ------------------------------------------------------------------ #

    def _track_squat_arm_hold(self, l_arm: float, r_arm: float):
        """Count frames where arm drops during an active squat rep."""
        if self._left.state == "ready" and self._right.state == "ready":
            return  # not in a rep
        self._squat_total_frames += 1
        l_drop = (l_arm == l_arm) and l_arm > SQUAT_ARM_HOLD_MAX  # NaN != NaN
        r_drop = (r_arm == r_arm) and r_arm > SQUAT_ARM_HOLD_MAX
        if l_drop or r_drop:
            self._squat_arm_drop_frames += 1

    def _tick_squat(self, l_angle: float, r_angle: float):
        for leg, angle in ((self._left, l_angle), (self._right, r_angle)):
            if angle != angle:   # NaN — leg not visible
                continue
            if leg.state == "ready" and angle < SQUAT_START:
                if self._rep_start_time is None:
                    self._rep_start_time = time.time()
            leg.tick_squat(angle)

    def _tick_lateral(self, l_angle: float, r_angle: float):
        for arm, angle in ((self._left, l_angle), (self._right, r_angle)):
            if angle != angle:
                continue
            if arm.state == "ready" and angle > LAT_START:
                if self._rep_start_time is None:
                    self._rep_start_time = time.time()
            arm.tick_lateral(angle, self.fast_rep_seconds)

    # ------------------------------------------------------------------ #
    # Sync check — fire a rep when both arms finish                        #
    # ------------------------------------------------------------------ #

    def _check_sync(self):
        left_done  = self._left.done
        right_done = self._right.done

        # Timeout: if one arm finished but the other hasn't within SYNC_WINDOW
        if self._rep_start_time is not None:
            elapsed = time.time() - self._rep_start_time
            if elapsed > SYNC_WINDOW and (left_done or right_done) and not (left_done and right_done):
                done_arm   = self._left  if left_done  else self._right
                other_arm  = self._right if left_done  else self._left
                # Single-leg/arm fallback: if the other limb never left ready
                # (occluded by camera angle), count the active limb's rep as valid.
                if other_arm.state == "ready" and other_arm.rep_started_at is None:
                    fallback_issues = list(done_arm.issues)
                    with self._lock:
                        ex = self._exercise
                    if (ex == "squat" and
                            self._squat_total_frames > 5 and
                            self._squat_arm_drop_frames / self._squat_total_frames > SQUAT_ARM_DROP_RATIO):
                        fallback_issues.append("arm_dropped")
                    self.total_attempts += 1
                    if not fallback_issues:
                        self.clean_reps += 1
                    self.publish_stats(last_seconds=done_arm.duration,
                                       best_angle=done_arm.best_angle,
                                       issue=",".join(fallback_issues))
                else:
                    self.total_attempts += 1
                    missing = "right_late" if left_done else "left_late"
                    self.publish_stats(last_seconds=done_arm.duration,
                                       best_angle=done_arm.best_angle,
                                       issue=missing)
                self._reset_rep()
                return

        if not (left_done and right_done):
            return

        # Both arms finished — combine issues
        all_issues = list(set(self._left.issues + self._right.issues))
        best_angle = (min(self._left.best_angle or 180, self._right.best_angle or 180)
                      if self._exercise in ("bicep_curl", "squat")
                      else max(self._left.best_angle or 0, self._right.best_angle or 0))
        duration = max(self._left.duration, self._right.duration)

        # Weighted squat: flag if arms dropped too often during the rep
        with self._lock:
            ex = self._exercise
        if (ex == "squat" and
                self._squat_total_frames > 5 and
                self._squat_arm_drop_frames / self._squat_total_frames > SQUAT_ARM_DROP_RATIO):
            all_issues.append("arm_dropped")

        self.total_attempts += 1
        if not all_issues:
            self.clean_reps += 1
        self.publish_stats(last_seconds=duration, best_angle=best_angle,
                           issue=",".join(all_issues))
        self._reset_rep()


def main():
    rospy.init_node("rep_counter_node")
    RepCounterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
