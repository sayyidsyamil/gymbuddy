#!/usr/bin/env python3
"""Logic-driven encouragement: every Nth rep, publish a short cheer."""

import random

import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import WorkoutStats

CHEERS = [
    "Nice work, keep going!",
    "Stay controlled.",
    "Looking solid.",
    "Halfway there.",
    "Strong rep.",
]


class MotivationNode:
    def __init__(self):
        self.every_n = max(1, int(rospy.get_param("~every_n_reps", 3)))
        self.pub = rospy.Publisher("/coaching_output", String, queue_size=4)
        rospy.Subscriber("/workout_stats", WorkoutStats, self.on_stats, queue_size=4)
        self._last_cheered = 0
        rospy.loginfo("motivation_node cheering every %d clean reps", self.every_n)

    def on_stats(self, msg: WorkoutStats):
        # Only fire on the workout_manager-annotated stream (target_reps != 0)
        # so we don't double-cheer when both publishers fire.
        if msg.target_reps == 0:
            return
        reps = msg.clean_reps
        if reps == 0 or reps == self._last_cheered:
            return
        if reps % self.every_n != 0:
            return
        self._last_cheered = reps
        text = random.choice(CHEERS)
        if msg.target_reps > 0:
            text += f" {reps} of {msg.target_reps}."
        self.pub.publish(String(data=text))


def main():
    rospy.init_node("motivation_node")
    MotivationNode()
    rospy.spin()


if __name__ == "__main__":
    main()
