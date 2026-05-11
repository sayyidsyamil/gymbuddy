#!/usr/bin/env python3
"""Serializes /coaching_output and /form_status into a single TTS queue (pyttsx3)."""

import queue
import threading
import time

import pyttsx3
import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import FormStatus

# Form-status codes that warrant speaking out loud (most are too noisy to read every frame).
SPEAK_FORM = {
    "back_rounded": "Watch your back.",
    "elbow_drift": "Keep your elbow still.",
    "depth_reached": None,  # silent — visual feedback is enough
}


class TextToSpeechNode:
    def __init__(self):
        rate = int(rospy.get_param("~rate", 175))
        volume = float(rospy.get_param("~volume", 1.0))
        self.min_repeat_seconds = float(rospy.get_param("~min_repeat_seconds", 4.0))

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        self.q: "queue.Queue[str]" = queue.Queue()
        self._last_spoken = {}  # text -> last timestamp

        rospy.Subscriber("/coaching_output", String, self.on_coaching, queue_size=8)
        rospy.Subscriber("/form_status", FormStatus, self.on_form, queue_size=8)

        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()
        rospy.loginfo("text_to_speech_node ready")

    def _enqueue(self, text: str):
        text = text.strip()
        if not text:
            return
        now = time.time()
        last = self._last_spoken.get(text, 0.0)
        if now - last < self.min_repeat_seconds:
            return
        self._last_spoken[text] = now
        self.q.put(text)

    def on_coaching(self, msg: String):
        self._enqueue(msg.data)

    def on_form(self, msg: FormStatus):
        phrase = SPEAK_FORM.get(msg.status)
        if phrase:
            self._enqueue(phrase)

    def _run_loop(self):
        while not rospy.is_shutdown():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except RuntimeError as exc:
                rospy.logwarn("pyttsx3 runtime error: %s", exc)


def main():
    rospy.init_node("text_to_speech_node")
    TextToSpeechNode()
    rospy.spin()


if __name__ == "__main__":
    main()
