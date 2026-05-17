#!/usr/bin/env python3
"""Captures frames from a USB/CSI camera and publishes them as CompressedImage."""

import cv2
import rospy
from sensor_msgs.msg import CompressedImage


def main():
    rospy.init_node("camera_input_node")

    device = rospy.get_param("~device", 0)
    width = rospy.get_param("~width", 960)
    height = rospy.get_param("~height", 540)
    rate_hz = rospy.get_param("~rate", 30.0)
    jpeg_quality = int(rospy.get_param("~jpeg_quality", 80))
    topic = rospy.get_param("~topic", "/raw_camera_frame")
    retry_seconds = float(rospy.get_param("~retry_seconds", 3.0))

    pub = rospy.Publisher(topic, CompressedImage, queue_size=2)

    cap = None

    def open_camera():
        camera = cv2.VideoCapture(device)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return camera

    rospy.loginfo("camera_input_node publishing %dx%d JPEG @ %.1f Hz on %s",
                  width, height, rate_hz, topic)

    rate = rospy.Rate(rate_hz)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    try:
        while not rospy.is_shutdown():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = open_camera()
                if not cap.isOpened():
                    rospy.logerr_throttle(
                        10.0,
                        "camera_input_node could not open device %s; retrying. "
                        "On macOS, enable Camera permission for the app running ROS.",
                        device,
                    )
                    rospy.sleep(retry_seconds)
                    continue
                rospy.loginfo("camera_input_node opened device %s", device)

            ok, frame = cap.read()
            if not ok:
                rospy.logwarn_throttle(2.0, "camera read failed")
                rate.sleep()
                continue

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                rate.sleep()
                continue

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "camera"
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            pub.publish(msg)
            rate.sleep()
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
