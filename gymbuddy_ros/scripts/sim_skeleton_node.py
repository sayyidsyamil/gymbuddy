#!/usr/bin/env python3
"""Publishes synthetic curl skeletons for end-to-end ROS testing without a camera."""

import math

import rospy

from gymbuddy_ros.msg import IntentUpdate, Landmark, Skeleton


RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16


class SimSkeletonNode:
    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate", 5.0))
        self.image_width = int(rospy.get_param("~image_width", 960))
        self.image_height = int(rospy.get_param("~image_height", 540))
        self.auto_start = bool(rospy.get_param("~auto_start", True))
        self.active = self.auto_start
        self.angles = [165, 150, 125, 105, 85, 105, 130, 150, 165]
        self.index = 0

        self.pub = rospy.Publisher("/skeleton_data", Skeleton, queue_size=2)
        self.intent_pub = rospy.Publisher("/intent_update", IntentUpdate, queue_size=2, latch=True)
        rospy.Subscriber("/intent_update", IntentUpdate, self.on_intent, queue_size=4)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.on_timer)

        if self.auto_start:
            rospy.Timer(rospy.Duration(0.5), self.publish_start, oneshot=True)
        rospy.loginfo("sim_skeleton_node ready (auto_start=%s)", self.auto_start)

    def publish_start(self, _event):
        msg = IntentUpdate()
        msg.header.stamp = rospy.Time.now()
        msg.action = "start_set"
        msg.value = 0
        msg.text = "simulated start"
        self.intent_pub.publish(msg)

    def on_intent(self, msg: IntentUpdate):
        if msg.action == "start_set":
            self.active = True
            self.index = 0
        elif msg.action in ("stop_set", "reset"):
            self.active = False

    def on_timer(self, _event):
        if not self.active:
            return
        angle = self.angles[self.index % len(self.angles)]
        self.index += 1
        self.pub.publish(self.make_skeleton(angle))

    def make_skeleton(self, angle_degrees):
        landmarks = [Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]

        shoulder_px = (480.0, 200.0)
        elbow_px = (480.0, 330.0)
        arm_len = 130.0
        theta = math.radians(angle_degrees)
        wrist_px = (
            elbow_px[0] + arm_len * math.sin(theta),
            elbow_px[1] - arm_len * math.cos(theta),
        )

        for index, point in (
            (RIGHT_SHOULDER, shoulder_px),
            (RIGHT_ELBOW, elbow_px),
            (RIGHT_WRIST, wrist_px),
        ):
            landmarks[index] = Landmark(
                x=point[0] / self.image_width,
                y=point[1] / self.image_height,
                z=0.0,
                visibility=0.99,
            )

        msg = Skeleton()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "sim"
        msg.image_width = self.image_width
        msg.image_height = self.image_height
        msg.landmarks = landmarks
        return msg


def main():
    rospy.init_node("sim_skeleton_node")
    SimSkeletonNode()
    rospy.spin()


if __name__ == "__main__":
    main()
