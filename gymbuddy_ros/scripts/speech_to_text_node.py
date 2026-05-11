#!/usr/bin/env python3
"""On /system_wake_state True, records audio with VAD and transcribes via faster-whisper."""

import threading
import time

import numpy as np
import rospy
import sounddevice as sd
from faster_whisper import WhisperModel
from std_msgs.msg import Bool, String

SAMPLE_RATE = 16000


class SpeechToTextNode:
    def __init__(self):
        model_size = rospy.get_param("~model_size", "tiny.en")
        compute_type = rospy.get_param("~compute_type", "int8")
        device = rospy.get_param("~device", "cpu")
        self.max_seconds = float(rospy.get_param("~max_seconds", 6.0))
        self.silence_seconds = float(rospy.get_param("~silence_seconds", 1.0))
        self.silence_rms = float(rospy.get_param("~silence_rms", 0.012))

        rospy.loginfo("speech_to_text_node loading faster-whisper '%s' on %s", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self.text_pub = rospy.Publisher("/user_speech_raw", String, queue_size=4)
        self.wake_pub = rospy.Publisher("/system_wake_state", Bool, queue_size=1, latch=True)

        self._busy = threading.Lock()
        rospy.Subscriber("/system_wake_state", Bool, self.on_wake, queue_size=1)
        rospy.loginfo("speech_to_text_node ready")

    def on_wake(self, msg: Bool):
        if not msg.data:
            return
        if not self._busy.acquire(blocking=False):
            return
        threading.Thread(target=self._capture_and_transcribe, daemon=True).start()

    def _capture_and_transcribe(self):
        try:
            audio = self._record_until_silence()
            if audio.size == 0:
                rospy.logwarn("speech_to_text captured no audio")
                return
            text = self._transcribe(audio)
            if text:
                rospy.loginfo("STT: %s", text)
                self.text_pub.publish(String(data=text))
        finally:
            # release wake-state so wake_word_node knows we're done
            self.wake_pub.publish(Bool(data=False))
            self._busy.release()

    def _record_until_silence(self):
        chunk_seconds = 0.1
        chunk_samples = int(SAMPLE_RATE * chunk_seconds)
        buf = []
        silence_for = 0.0
        elapsed = 0.0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=chunk_samples) as stream:
            while elapsed < self.max_seconds and not rospy.is_shutdown():
                data, _ = stream.read(chunk_samples)
                samples = data[:, 0]
                buf.append(samples.copy())
                rms = float(np.sqrt(np.mean(samples ** 2)))
                if rms < self.silence_rms:
                    silence_for += chunk_seconds
                    if silence_for >= self.silence_seconds and elapsed > 0.4:
                        break
                else:
                    silence_for = 0.0
                elapsed += chunk_seconds
        if not buf:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(buf)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()


def main():
    rospy.init_node("speech_to_text_node")
    SpeechToTextNode()
    rospy.spin()


if __name__ == "__main__":
    main()
