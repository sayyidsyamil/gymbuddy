#!/usr/bin/env python3
"""Camera bridge: publishes /raw_camera_frame (sensor_msgs/CompressedImage).

Two sources, chosen by `~source`:
  usb   — open a V4L/USB device via OpenCV  (~device, default 0)
  topic — subscribe to a sensor_msgs/Image  (~input_topic,
          default /camera/color/image_raw — the Astra colour stream)

Frames are JPEG-encoded and republished on `~topic` (default /raw_camera_frame).
If `~width`/`~height` are set, frames are resized so the downstream pipeline
(pose detection, display) sees a consistent resolution.
"""

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image


class CameraInputNode:
    def __init__(self):
        self.source        = rospy.get_param("~source",        "usb").lower()
        self.width         = int(rospy.get_param("~width",       960))
        self.height        = int(rospy.get_param("~height",      540))
        self.rate_hz       = float(rospy.get_param("~rate",       30.0))
        self.jpeg_quality  = int(rospy.get_param("~jpeg_quality", 80))
        self.topic         = rospy.get_param("~topic",         "/raw_camera_frame")
        self.input_topic   = rospy.get_param("~input_topic",   "/camera/color/image_raw")
        self.device        = rospy.get_param("~device",          0)
        self.retry_seconds = float(rospy.get_param("~retry_seconds", 3.0))
        # Mirror the frame so a user-facing webcam shows "selfie" view and the
        # skeleton's left/right matches the on-screen left/right.
        self.flip_horizontal = bool(rospy.get_param("~flip_horizontal", True))
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        self.pub = rospy.Publisher(self.topic, CompressedImage, queue_size=2)

        if self.source == "topic":
            self.bridge = CvBridge()
            rospy.Subscriber(self.input_topic, Image, self._on_image, queue_size=1)
            rospy.loginfo("camera_input_node bridging %s -> %s (JPEG q=%d, %dx%d)",
                          self.input_topic, self.topic,
                          self.jpeg_quality, self.width, self.height)

    def _resize_if_needed(self, bgr):
        if self.width and self.height and \
                (bgr.shape[1] != self.width or bgr.shape[0] != self.height):
            return cv2.resize(bgr, (self.width, self.height))
        return bgr

    def _maybe_flip(self, bgr):
        if self.flip_horizontal:
            return cv2.flip(bgr, 1)
        return bgr

    def _publish(self, bgr, stamp=None):
        bgr = self._maybe_flip(bgr)
        ok, buf = cv2.imencode(".jpg", bgr, self.encode_params)
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = "camera"
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self.pub.publish(msg)

    def _on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "imgmsg_to_cv2 failed: %s", exc)
            return
        self._publish(self._resize_if_needed(bgr), stamp=msg.header.stamp)

    def run_usb_loop(self):
        rospy.loginfo("camera_input_node publishing %dx%d JPEG @ %.1f Hz on %s (device=%s)",
                      self.width, self.height, self.rate_hz, self.topic, self.device)
        rate = rospy.Rate(self.rate_hz)
        cap = None
        try:
            while not rospy.is_shutdown():
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(self.device)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    if not cap.isOpened():
                        rospy.logerr_throttle(
                            10.0,
                            "camera_input_node could not open device %s; retrying.",
                            self.device,
                        )
                        rospy.sleep(self.retry_seconds)
                        continue
                    rospy.loginfo("camera_input_node opened device %s", self.device)

                ok, frame = cap.read()
                if not ok:
                    rospy.logwarn_throttle(2.0, "camera read failed")
                    rate.sleep()
                    continue
                self._publish(self._resize_if_needed(frame))
                rate.sleep()
        finally:
            if cap is not None:
                cap.release()


def main():
    rospy.init_node("camera_input_node")
    node = CameraInputNode()
    if node.source == "usb":
        node.run_usb_loop()
    else:
        rospy.spin()


if __name__ == "__main__":
    main()
