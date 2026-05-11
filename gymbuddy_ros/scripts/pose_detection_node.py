#!/usr/bin/env python3
"""Runs MediaPipe Pose on incoming frames and publishes 33 landmarks."""

import cv2
import numpy as np
import rospy
import mediapipe as mp
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray, MultiArrayDimension

from gymbuddy_ros.msg import Landmark, Skeleton


class PoseDetectionNode:
    def __init__(self):
        model_complexity = int(rospy.get_param("~model_complexity", 1))
        min_detection_conf = float(rospy.get_param("~min_detection_confidence", 0.5))
        min_tracking_conf = float(rospy.get_param("~min_tracking_confidence", 0.5))

        self.skeleton_pub = rospy.Publisher("/skeleton_data", Skeleton, queue_size=2)
        self.bbox_pub = rospy.Publisher("/target_object_bbox", Int32MultiArray, queue_size=2)

        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            enable_segmentation=False,
        )

        rospy.Subscriber("/raw_camera_frame", CompressedImage, self.on_frame, queue_size=1)
        rospy.loginfo("pose_detection_node ready")

    def on_frame(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return

        skel = Skeleton()
        skel.header = msg.header
        skel.image_width = w
        skel.image_height = h

        xs, ys = [], []
        for lm in result.pose_landmarks.landmark:
            skel.landmarks.append(Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility))
            xs.append(lm.x * w)
            ys.append(lm.y * h)

        self.skeleton_pub.publish(skel)

        if xs and ys:
            bbox = Int32MultiArray()
            bbox.layout.dim = [MultiArrayDimension(label="bbox", size=4, stride=4)]
            bbox.data = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            self.bbox_pub.publish(bbox)


def main():
    rospy.init_node("pose_detection_node")
    PoseDetectionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
