#!/usr/bin/env python3
"""Elbow-angle state machine that increments rep count from /form_status."""

import time

import rospy

from gymbuddy_ros.msg import FormStatus, IntentUpdate, WorkoutStats

EXTENDED_ANGLE = 155.0
CURLING_ANGLE = 135.0
TOP_ANGLE = 90.0
PARTIAL_TOP_ANGLE = 120.0
FAST_REP_SECONDS = 0.6


class RepCounterNode:
    def __init__(self):
        self.extended_angle = float(rospy.get_param("~extended_angle", EXTENDED_ANGLE))
        self.curling_angle = float(rospy.get_param("~curling_angle", CURLING_ANGLE))
        self.top_angle = float(rospy.get_param("~top_angle", TOP_ANGLE))
        self.partial_top_angle = float(rospy.get_param("~partial_top_angle", PARTIAL_TOP_ANGLE))
        self.fast_rep_seconds = float(rospy.get_param("~fast_rep_seconds", FAST_REP_SECONDS))
        self.pub = rospy.Publisher("/workout_stats", WorkoutStats, queue_size=4, latch=True)
        rospy.Subscriber("/form_status", FormStatus, self.on_form, queue_size=4)
        rospy.Subscriber("/intent_update", IntentUpdate, self.on_intent, queue_size=4)

        self.active = False
        self.state = "ready"
        self.clean_reps = 0
        self.total_attempts = 0
        self.rep_started_at = None
        self.min_angle = 180.0
        self.saw_top = False
        self.saw_partial = False
        rospy.loginfo("rep_counter_node ready")

    def reset_set(self):
        self.state = "ready"
        self.clean_reps = 0
        self.total_attempts = 0
        self.reset_rep()
        self.publish_stats()

    def reset_rep(self):
        self.rep_started_at = None
        self.min_angle = 180.0
        self.saw_top = False
        self.saw_partial = False

    def publish_stats(self, last_seconds=0.0, last_min_angle=180.0, issue=""):
        msg = WorkoutStats()
        msg.header.stamp = rospy.Time.now()
        msg.exercise = "one_arm_bicep_curl"
        msg.clean_reps = self.clean_reps
        msg.total_attempts = self.total_attempts
        msg.target_reps = 0  # workout_manager owns the target; this is informational
        msg.last_rep_seconds = float(last_seconds)
        msg.last_min_angle = float(last_min_angle)
        msg.last_rep_issue = issue
        self.pub.publish(msg)

    def on_intent(self, msg: IntentUpdate):
        if msg.action == "start_set":
            self.active = True
            self.reset_set()
            rospy.loginfo("rep_counter_node set started")
        elif msg.action == "stop_set":
            self.active = False
            self.reset_rep()
            self.state = "ready"
            rospy.loginfo("rep_counter_node set stopped")
        elif msg.action == "reset":
            self.active = False
            self.reset_set()
            rospy.loginfo("rep_counter_node reset")

    def finish_rep(self, full_extension: bool):
        if self.rep_started_at is None:
            self.reset_rep()
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
                           last_min_angle=self.min_angle,
                           issue=",".join(issues))
        self.reset_rep()
        self.state = "ready"

    def on_form(self, msg: FormStatus):
        if not self.active:
            return
        if msg.status == "arm_not_visible":
            return
        angle = msg.elbow_angle
        if angle != angle:  # NaN
            return
        self.min_angle = min(self.min_angle, angle)

        if self.state == "ready":
            if angle < self.curling_angle:
                self.rep_started_at = time.time()
                self.min_angle = angle
                self.saw_top = False
                self.saw_partial = False
                self.state = "curling"
        elif self.state == "curling":
            if angle <= self.top_angle:
                self.saw_top = True
                self.state = "top"
            elif angle <= self.partial_top_angle:
                self.saw_partial = True
            elif angle >= self.extended_angle and (self.saw_partial or self.min_angle < self.curling_angle):
                self.finish_rep(full_extension=True)
        elif self.state == "top":
            if angle > self.top_angle + 15:
                self.state = "lowering"
        elif self.state == "lowering":
            if angle >= self.extended_angle:
                self.finish_rep(full_extension=True)
            elif angle < self.curling_angle - 10:
                self.finish_rep(full_extension=False)


def main():
    rospy.init_node("rep_counter_node")
    RepCounterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
