#!/usr/bin/env python3
"""On /system_wake_state True, records audio with VAD and transcribes via faster-whisper.

faster-whisper expects 16 kHz float32 mono. If the device doesn't support 16 kHz
natively (common with USB mics), we open it at the device's native rate and
resample to 16 kHz before passing to Whisper.
"""

import threading
import time

import numpy as np
import rospy
import sounddevice as sd
from faster_whisper import WhisperModel
from std_msgs.msg import Bool, String

TARGET_RATE = 16000


def _resolve_audio_device(raw):
    """Accept None, '', -1, an int index, or a (sub)string device name."""
    if raw is None or raw == "" or raw == -1 or raw == "-1":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


def _device_sample_rate(device):
    info = sd.query_devices(device, "input")
    return int(info["default_samplerate"])


class SpeechToTextNode:
    def __init__(self):
        model_size = rospy.get_param("~model_size", "tiny.en")
        compute_type = rospy.get_param("~compute_type", "int8")
        device = rospy.get_param("~device", "cpu")
        self.max_seconds = float(rospy.get_param("~max_seconds", 6.0))
        self.silence_seconds = float(rospy.get_param("~silence_seconds", 1.0))
        self.silence_rms = float(rospy.get_param("~silence_rms", 0.012))
        self.audio_device = _resolve_audio_device(rospy.get_param("~audio_device", -1))

        rospy.loginfo("speech_to_text_node loading faster-whisper '%s' on %s", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self.device_rate = self._select_device_rate()
        self.needs_resample = self.device_rate != TARGET_RATE

        self.text_pub = rospy.Publisher("/user_speech_raw", String, queue_size=4)
        self.wake_pub = rospy.Publisher("/system_wake_state", Bool, queue_size=1, latch=True)

        self._busy = threading.Lock()
        rospy.Subscriber("/system_wake_state", Bool, self.on_wake, queue_size=1)
        rospy.loginfo(
            "speech_to_text_node ready (device=%s, rate=%d Hz%s)",
            self.audio_device, self.device_rate,
            " → 16 kHz resampled" if self.needs_resample else "",
        )

    def _select_device_rate(self):
        try:
            sd.check_input_settings(device=self.audio_device, samplerate=TARGET_RATE,
                                    channels=1, dtype="float32")
            return TARGET_RATE
        except Exception:
            rate = _device_sample_rate(self.audio_device)
            rospy.logwarn(
                "speech_to_text_node: device %s does not support 16 kHz; using %d Hz with resampling",
                self.audio_device, rate,
            )
            return rate

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
            if self.needs_resample:
                audio = self._resample_to_16k(audio)
            text = self._transcribe(audio)
            if text:
                rospy.loginfo("STT: %s", text)
                self.text_pub.publish(String(data=text))
        finally:
            self.wake_pub.publish(Bool(data=False))
            self._busy.release()

    def _resample_to_16k(self, samples: np.ndarray) -> np.ndarray:
        """Linear resample float32 mono → 16 kHz."""
        if samples.size == 0:
            return samples
        out_len = int(round(samples.size * TARGET_RATE / float(self.device_rate)))
        if out_len <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, out_len,       endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)

    def _record_until_silence(self):
        chunk_seconds = 0.1
        chunk_samples = max(1, int(self.device_rate * chunk_seconds))
        buf = []
        silence_for = 0.0
        elapsed = 0.0
        with sd.InputStream(samplerate=self.device_rate, channels=1, dtype="float32",
                            blocksize=chunk_samples,
                            device=self.audio_device) as stream:
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
