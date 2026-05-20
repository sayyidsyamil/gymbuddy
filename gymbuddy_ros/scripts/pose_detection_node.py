#!/usr/bin/env python3


import cv2
import mediapipe as mp
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray, MultiArrayDimension

from gymbuddy_ros.msg import Landmark, Skeleton

BLAZE_TO_COCO = [
    None,  # 0  nose          — skipped
    None,  # 1  left eye      — skipped
    None,  # 2  right eye     — skipped
    None,  # 3  left ear      — skipped
    None,  # 4  right ear     — skipped
    11,    # 5  left shoulder
    12,    # 6  right shoulder
    13,    # 7  left elbow
    14,    # 8  right elbow
    15,    # 9  left wrist
    16,    # 10 right wrist
    23,    # 11 left hip
    24,    # 12 right hip
    25,    # 13 left knee
    26,    # 14 right knee
    27,    # 15 left ankle
    28,    # 16 right ankle
]

# MediaPipe expects a reasonably large input. Upscaling small frames before
INFERENCE_SIDE = 640


class PoseDetectionNode:
    def __init__(self):
        # complexity 1 = "full" — best speed/accuracy balance on CPU.
        # 0 = "lite" (fastest, lower accuracy), 2 = "heavy" (slow on CPU).
        model_complexity = int(rospy.get_param("~model_complexity", 1))
        min_detection_conf = float(rospy.get_param("~min_detection_confidence", 0.6))
        min_tracking_conf = float(rospy.get_param("~min_tracking_confidence", 0.6))
        # Smoothing reduces jitter between frames.
        smooth_landmarks = bool(rospy.get_param("~smooth_landmarks", True))

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

        self.skeleton_pub = rospy.Publisher("/skeleton_data",      Skeleton,         queue_size=2)
        self.bbox_pub     = rospy.Publisher("/target_object_bbox", Int32MultiArray,  queue_size=2)

        rospy.Subscriber("/raw_camera_frame", CompressedImage, self.on_frame, queue_size=1)
        rospy.loginfo(
            "pose_detection_node ready (MediaPipe Pose, complexity=%d, smooth=%s, body-only)",
            model_complexity, smooth_landmarks,
        )

    def on_frame(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        h, w = bgr.shape[:2]

        # Upscale small frames so the model has more pixels to work with.
        long_side = max(h, w)
        if long_side < INFERENCE_SIDE:
            scale = INFERENCE_SIDE / float(long_side)
            inf_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            inf_bgr = bgr

        rgb = cv2.cvtColor(inf_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        if result.pose_landmarks is None:
            return

        blaze = result.pose_landmarks.landmark  # 33 entries

        skel = Skeleton()
        skel.header       = msg.header
        skel.image_width  = w
        skel.image_height = h

        xs, ys = [], []
        for blaze_idx in BLAZE_TO_COCO:
            if blaze_idx is None:
                # face slot — keep array length stable, but visibility=0 means
                # downstream nodes skip it.
                skel.landmarks.append(Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0))
                continue
            lm = blaze[blaze_idx]
            skel.landmarks.append(Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            ))
            if lm.visibility >= 0.3:
                xs.append(lm.x * w)
                ys.append(lm.y * h)

        self.skeleton_pub.publish(skel)

        if xs and ys:
            x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
            x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))
            bbox = Int32MultiArray()
            bbox.layout.dim = [MultiArrayDimension(label="bbox", size=4, stride=4)]
            bbox.data = [x1, y1, x2, y2]
            self.bbox_pub.publish(bbox)


def main():
    rospy.init_node("pose_detection_node")
    PoseDetectionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
