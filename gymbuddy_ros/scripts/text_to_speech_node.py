#!/usr/bin/env python3
"""TTS node — speaks only on two conditions:

  1. /tts_priority  — always spoken (motivation cheers every 35 reps).
  2. /coaching_output — spoken only within a gate window that opens for
     `wake_gate_seconds` after the wake word fires (/system_wake_state True).
     This covers the LLM reply and workout-manager confirmations that follow
     a voice command, while silencing the constant per-rep auto-coaching.
"""

import queue
import threading
import time

import pyttsx3
import rospy
from std_msgs.msg import Bool, String


class TextToSpeechNode:
    def __init__(self):
        rate      = int(rospy.get_param("~rate",              175))
        volume    = float(rospy.get_param("~volume",           1.0))
        self.min_repeat_seconds  = float(rospy.get_param("~min_repeat_seconds", 4.0))
        self.wake_gate_seconds   = float(rospy.get_param("~wake_gate_seconds",  30.0))

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        self._q: "queue.Queue[str]" = queue.Queue()
        self._last_spoken: "dict[str, float]" = {}
        self._gate_until = 0.0   # epoch time; coaching_output allowed before this

        rospy.Subscriber("/system_wake_state", Bool,   self._on_wake,     queue_size=1)
        rospy.Subscriber("/coaching_output",   String, self._on_coaching, queue_size=8)
        rospy.Subscriber("/tts_priority",      String, self._on_priority, queue_size=8)

        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()
        rospy.loginfo("text_to_speech_node ready (wake gate=%.0fs, motivation@/tts_priority)",
                      self.wake_gate_seconds)

    def _enqueue(self, text: str):
        text = text.strip()
        if not text:
            return
        now = time.time()
        if now - self._last_spoken.get(text, 0.0) < self.min_repeat_seconds:
            return
        self._last_spoken[text] = now
        self._q.put(text)

    def _on_wake(self, msg: Bool):
        if msg.data:
            self._gate_until = time.time() + self.wake_gate_seconds
            rospy.logdebug("TTS gate opened for %.0f s", self.wake_gate_seconds)

    def _on_coaching(self, msg: String):
        if time.time() < self._gate_until:
            self._enqueue(msg.data)

    def _on_priority(self, msg: String):
        self._enqueue(msg.data)

    def _run_loop(self):
        while not rospy.is_shutdown():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except RuntimeError as exc:
                rospy.logwarn("pyttsx3 error: %s", exc)


def main():
    rospy.init_node("text_to_speech_node")
    TextToSpeechNode()
    rospy.spin()


if __name__ == "__main__":
    main()
