#!/usr/bin/env python3
"""Central state machine for the active exercise, set history, and target reps."""

import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import IntentUpdate, WorkoutStats


class WorkoutManagerNode:
    def __init__(self):
        self.target_reps = int(rospy.get_param("~initial_target", 10))
        self.exercise = "one_arm_bicep_curl"
        self.set_active = False
        self.history = []  # list of completed-set summaries

        self.coach_pub = rospy.Publisher("/coaching_output", String, queue_size=4, latch=True)
        self.stats_pub = rospy.Publisher("/workout_stats", WorkoutStats, queue_size=4, latch=True)
        self.intent_pub = rospy.Publisher("/intent_update", IntentUpdate, queue_size=4)

        rospy.Subscriber("/workout_stats", WorkoutStats, self.on_stats, queue_size=4)
        rospy.Subscriber("/intent_update", IntentUpdate, self.on_intent, queue_size=4)
        rospy.loginfo("workout_manager_node ready (target=%d)", self.target_reps)

    def on_intent(self, msg: IntentUpdate):
        if msg.action == "start_set":
            self.set_active = True
            self.coach_pub.publish(String(data=f"Starting your set. Target {self.target_reps} reps."))
        elif msg.action == "stop_set":
            self.set_active = False
            self.coach_pub.publish(String(data="Set ended."))
        elif msg.action == "reset":
            self.history.clear()
            self.coach_pub.publish(String(data="Workout reset."))
        elif msg.action == "update_target":
            self.target_reps = max(1, msg.value)
            self.coach_pub.publish(String(data=f"Target updated to {self.target_reps} reps."))
        elif msg.action == "add_reps":
            self.target_reps += max(0, msg.value)
            self.coach_pub.publish(String(data=f"Target now {self.target_reps} reps."))
        elif msg.action == "remove_reps":
            self.target_reps = max(1, self.target_reps - msg.value)
            self.coach_pub.publish(String(data=f"Target reduced to {self.target_reps} reps."))

    def on_stats(self, msg: WorkoutStats):
        # Avoid feedback loop: only act on stats with target_reps == 0 (from rep_counter).
        if msg.target_reps != 0:
            return

        if self.set_active and msg.clean_reps >= self.target_reps:
            self.set_active = False
            self.history.append({
                "exercise": msg.exercise,
                "clean_reps": msg.clean_reps,
                "total_attempts": msg.total_attempts,
            })
            self.coach_pub.publish(
                String(data=f"Target hit: {msg.clean_reps} of {msg.total_attempts}. Set complete.")
            )
            stop = IntentUpdate()
            stop.header.stamp = rospy.Time.now()
            stop.action = "stop_set"
            stop.value = 0
            stop.text = "target hit"
            self.intent_pub.publish(stop)

        # Re-emit stats annotated with the manager's target so consumers see the goal.
        out = WorkoutStats()
        out.header.stamp = rospy.Time.now()
        out.exercise = msg.exercise
        out.clean_reps = msg.clean_reps
        out.total_attempts = msg.total_attempts
        out.target_reps = self.target_reps
        out.last_rep_seconds = msg.last_rep_seconds
        out.last_min_angle = msg.last_min_angle
        out.last_rep_issue = msg.last_rep_issue
        self.stats_pub.publish(out)


def main():
    rospy.init_node("workout_manager_node")
    WorkoutManagerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
