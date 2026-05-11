#!/usr/bin/env python3
"""Listens for a wake word via openwakeword and toggles /system_wake_state."""

import threading

import numpy as np
import rospy
import sounddevice as sd
from openwakeword.model import Model
from std_msgs.msg import Bool

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # openwakeword expects ~80 ms @ 16 kHz


class WakeWordNode:
    def __init__(self):
        # openwakeword ships with "hey_jarvis", "alexa", "hey_mycroft" — closest to "Hey Buddy"
        self.wake_model_name = rospy.get_param("~wake_model", "hey_jarvis")
        self.threshold = float(rospy.get_param("~threshold", 0.5))
        self.cooldown_seconds = float(rospy.get_param("~cooldown_seconds", 8.0))

        self.pub = rospy.Publisher("/system_wake_state", Bool, queue_size=1, latch=True)
        self.pub.publish(Bool(data=False))

        rospy.loginfo("wake_word_node loading openwakeword model: %s", self.wake_model_name)
        self.model = Model(wakeword_models=[self.wake_model_name])

        self._cooldown_until = rospy.Time(0)
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=self._on_audio,
        )
        self._stream.start()
        rospy.loginfo("wake_word_node listening for '%s'", self.wake_model_name)

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            rospy.logwarn_throttle(5.0, "wake_word audio status: %s", status)
        if rospy.Time.now() < self._cooldown_until:
            return
        samples = np.frombuffer(indata.tobytes(), dtype=np.int16)
        scores = self.model.predict(samples)
        for name, score in scores.items():
            if score >= self.threshold:
                with self._lock:
                    if rospy.Time.now() < self._cooldown_until:
                        return
                    self._cooldown_until = rospy.Time.now() + rospy.Duration.from_sec(self.cooldown_seconds)
                rospy.loginfo("wake word '%s' triggered (score=%.2f)", name, score)
                self.pub.publish(Bool(data=True))
                return

    def shutdown(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


def main():
    rospy.init_node("wake_word_node")
    node = WakeWordNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
