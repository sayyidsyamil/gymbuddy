#!/usr/bin/env python3
"""Exercise-aware rep counter.

Bicep curl  — tracks elbow flexion angle; issues: partial_curl, incomplete_extension, too_fast
Lateral raise — tracks shoulder abduction angle; issues: partial_raise, no_return, too_fast
"""

import threading
import time

import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import FormStatus, IntentUpdate, WorkoutStats

# Bicep curl thresholds
# 145° is a realistic "arm mostly straight" — most people don't lock out to 155°
# between reps and we don't want to mark every clean rep as incomplete_extension.
CURL_EXTENDED      = 145.0
CURL_CURLING       = 130.0
CURL_TOP           = 95.0   # ~95° still counts as "reached the top"
CURL_PARTIAL_TOP   = 115.0

# Lateral raise thresholds
LAT_DOWN           = 30.0
LAT_START          = 32.0   # slight hysteresis to avoid noise triggering
LAT_PARTIAL        = 55.0
LAT_TOP            = 75.0

FAST_REP_SECONDS   = 0.6


class RepCounterNode:
    def __init__(self):
        # curl params (ROS params kept for backwards compat)
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
        rospy.loginfo("rep_counter_node ready")

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
    # Shared rep state                                                     #
    # ------------------------------------------------------------------ #

    def _reset_counters(self):
        self.state          = "ready"
        self.clean_reps     = 0
        self.total_attempts = 0
        self._reset_rep()

    def _reset_rep(self):
        self.rep_started_at = None
        self.best_angle     = None   # min for curl, max for lateral raise
        self.saw_top        = False
        self.saw_partial    = False

    def publish_stats(self, last_seconds=0.0, best_angle=None, issue=""):
        with self._lock:
            exercise = self._exercise
        if best_angle is None:
            best_angle = 180.0 if exercise == "bicep_curl" else 0.0
        msg = WorkoutStats()
        msg.header.stamp    = rospy.Time.now()
        msg.exercise        = exercise
        msg.clean_reps      = self.clean_reps
        msg.total_attempts  = self.total_attempts
        msg.target_reps     = 0   # workout_manager owns the target
        msg.last_rep_seconds = float(last_seconds)
        msg.last_min_angle  = float(best_angle)
        msg.last_rep_issue  = issue
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
            self.state = "ready"
            rospy.loginfo("rep_counter: set stopped")
        elif msg.action == "reset":
            self.active = False
            self._reset_counters()
            self.publish_stats()
            rospy.loginfo("rep_counter: reset")

    # ------------------------------------------------------------------ #
    # Form callback — dispatch per exercise                                #
    # ------------------------------------------------------------------ #

    def on_form(self, msg: FormStatus):
        if not self.active:
            return
        if msg.status == "arm_not_visible":
            return
        angle = msg.elbow_angle
        if angle != angle:   # NaN
            return
        with self._lock:
            exercise = self._exercise
        if exercise == "lateral_raise":
            self._on_form_lateral(angle)
        else:
            self._on_form_curl(angle)

    # ------------------------------------------------------------------ #
    # Bicep curl state machine                                             #
    # ------------------------------------------------------------------ #

    def _on_form_curl(self, angle: float):
        if self.best_angle is None:
            self.best_angle = angle
        self.best_angle = min(self.best_angle, angle)

        if self.state == "ready":
            if angle < self.curling_angle:
                self.rep_started_at = time.time()
                self.best_angle     = angle
                self.saw_top = self.saw_partial = False
                self.state = "curling"

        elif self.state == "curling":
            if angle <= self.top_angle:
                self.saw_top = True
                self.state = "top"
            elif angle <= self.partial_top_angle:
                self.saw_partial = True
            elif angle >= self.extended_angle and (self.saw_partial or self.best_angle < self.curling_angle):
                self._finish_curl(full_extension=True)

        elif self.state == "top":
            if angle > self.top_angle + 15:
                self.state = "lowering"

        elif self.state == "lowering":
            if angle >= self.extended_angle:
                self._finish_curl(full_extension=True)
            elif angle < self.curling_angle - 10:
                self._finish_curl(full_extension=False)

    def _finish_curl(self, full_extension: bool):
        if self.rep_started_at is None:
            self._reset_rep()
            self.state = "ready"
            return
        duration = time.time() - self.rep_started_at
        issues = []
        if not self.saw_top:
            issues.append("partial_curl")
        if not full_extension:
            issues.append("incomplete_extension")
        if duration < self.fast_rep_seconds:
            issues.append("too_fast")
        self.total_attempts += 1
        if not issues:
            self.clean_reps += 1
        self.publish_stats(last_seconds=duration,
                           best_angle=self.best_angle,
                           issue=",".join(issues))
        self._reset_rep()
        self.state = "ready"

    # ------------------------------------------------------------------ #
    # Lateral raise state machine                                          #
    # ------------------------------------------------------------------ #

    def _on_form_lateral(self, angle: float):
        if self.best_angle is None:
            self.best_angle = angle
        self.best_angle = max(self.best_angle, angle)   # track highest point

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
                # arm dropped back without reaching top — incomplete attempt
                self._finish_lateral(returned=True)

        elif self.state == "at_top":
            if angle < LAT_TOP - 10:
                self.state = "lowering"

        elif self.state == "lowering":
            if angle <= LAT_DOWN:
                self._finish_lateral(returned=True)
            elif angle >= LAT_TOP:
                self.state = "at_top"   # raised again

    def _finish_lateral(self, returned: bool):
        if self.rep_started_at is None:
            self._reset_rep()
            self.state = "ready"
            return
        duration = time.time() - self.rep_started_at
        issues = []
        if not self.saw_top:
            issues.append("partial_raise")
        if not returned:
            issues.append("no_return")
        if duration < self.fast_rep_seconds:
            issues.append("too_fast")
        self.total_attempts += 1
        if not issues:
            self.clean_reps += 1
        self.publish_stats(last_seconds=duration,
                           best_angle=self.best_angle,
                           issue=",".join(issues))
        self._reset_rep()
        self.state = "ready"


def main():
    rospy.init_node("rep_counter_node")
    RepCounterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
