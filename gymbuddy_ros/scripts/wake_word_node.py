#!/usr/bin/env python3
"""Listens for a wake word via openwakeword and toggles /system_wake_state.

openWakeWord expects 16 kHz int16 mono. Many USB mics only support 44.1/48 kHz,
so we open the stream at the device's native rate and resample to 16 kHz in the
callback (linear interpolation — fine for speech).
"""

import threading

import numpy as np
import rospy
import sounddevice as sd
from openwakeword.model import Model
from std_msgs.msg import Bool, String

TARGET_RATE   = 16000
CHUNK_MS      = 80              # openwakeword wants 80 ms blocks
CHUNK_AT_16K  = 1280            # 80 ms @ 16 kHz


def _resolve_audio_device(raw):
    """Accept None, '', -1, an int index, or a (sub)string device name."""
    if raw is None or raw == "" or raw == -1 or raw == "-1":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw  # let sounddevice match by name


def _device_sample_rate(device):
    info = sd.query_devices(device, "input")
    return int(info["default_samplerate"])


class WakeWordNode:
    def __init__(self):
        self.wake_model_name = rospy.get_param("~wake_model", "hey_jarvis")
        self.threshold = float(rospy.get_param("~threshold", 0.5))
        self.cooldown_seconds = float(rospy.get_param("~cooldown_seconds", 8.0))
        self.audio_device = _resolve_audio_device(rospy.get_param("~audio_device", -1))
        # Spoken prompt + delay before STT starts. Delay must cover the time it
        # takes pyttsx3 to say the prompt, or the mic will catch the tail end.
        self.prompt_text = rospy.get_param("~prompt_text", "What can I help you with?")
        self.prompt_delay_seconds = float(rospy.get_param("~prompt_delay_seconds", 1.6))

        self.pub = rospy.Publisher("/system_wake_state", Bool, queue_size=1, latch=True)
        self.prompt_pub = rospy.Publisher("/tts_priority", String, queue_size=2)
        self.pub.publish(Bool(data=False))

        rospy.loginfo("wake_word_node loading openwakeword model: %s", self.wake_model_name)
        self.model = Model(wakeword_models=[self.wake_model_name], inference_framework="onnx")

        # Pick a sample rate the device actually supports.
        self.device_rate = self._select_device_rate()
        self.chunk_samples = max(1, int(self.device_rate * CHUNK_MS / 1000))
        self.needs_resample = self.device_rate != TARGET_RATE

        self._cooldown_until = rospy.Time(0)
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=self.device_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_samples,
            callback=self._on_audio,
            device=self.audio_device,
        )
        self._stream.start()
        rospy.loginfo(
            "wake_word_node listening for '%s' (device=%s, rate=%d Hz%s)",
            self.wake_model_name, self.audio_device, self.device_rate,
            " → 16 kHz resampled" if self.needs_resample else "",
        )

    def _select_device_rate(self):
        """Prefer 16 kHz if the device supports it; otherwise use the device's native rate."""
        try:
            sd.check_input_settings(device=self.audio_device, samplerate=TARGET_RATE,
                                    channels=1, dtype="int16")
            return TARGET_RATE
        except Exception:
            rate = _device_sample_rate(self.audio_device)
            rospy.logwarn(
                "wake_word_node: device %s does not support 16 kHz; using %d Hz with resampling",
                self.audio_device, rate,
            )
            return rate

    def _resample_to_16k(self, samples: np.ndarray) -> np.ndarray:
        """Linear resample int16 mono → 16 kHz. Cheap and good enough for speech."""
        if not self.needs_resample or samples.size == 0:
            return samples
        out_len = int(round(samples.size * TARGET_RATE / float(self.device_rate)))
        if out_len <= 0:
            return np.zeros(0, dtype=np.int16)
        # interp expects float; clip back to int16 at the end
        x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, out_len,       endpoint=False)
        resampled = np.interp(x_new, x_old, samples.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            rospy.logwarn_throttle(5.0, "wake_word audio status: %s", status)
        if rospy.Time.now() < self._cooldown_until:
            return
        samples = np.frombuffer(indata.tobytes(), dtype=np.int16)
        samples = self._resample_to_16k(samples)
        if samples.size == 0:
            return
        scores = self.model.predict(samples)
        for name, score in scores.items():
            if score >= self.threshold:
                with self._lock:
                    if rospy.Time.now() < self._cooldown_until:
                        return
                    self._cooldown_until = rospy.Time.now() + rospy.Duration.from_sec(self.cooldown_seconds)
                rospy.loginfo("wake word '%s' triggered (score=%.2f)", name, score)
                # Offload prompt + delayed wake-flip to a worker — never block
                # the PortAudio callback.
                threading.Thread(target=self._handle_trigger,
                                 args=(name, float(score)), daemon=True).start()
                return

    def _handle_trigger(self, name: str, score: float):
        """Speak the prompt, wait for it to finish, then arm STT."""
        # Speak the prompt first (bypasses the TTS wake gate via /tts_priority).
        if self.prompt_text:
            self.prompt_pub.publish(String(data=self.prompt_text))
        # Give pyttsx3 time to finish before STT opens the mic.
        if self.prompt_delay_seconds > 0:
            rospy.sleep(self.prompt_delay_seconds)
        # Now signal STT to start recording.
        self.pub.publish(Bool(data=True))

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
