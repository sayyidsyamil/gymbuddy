#!/usr/bin/env python3
"""voice_io_node — Groq STT (Whisper) + Groq TTS (Orpheus).

Push-to-talk: hold SPACE to record, release to transcribe and publish.
Replaces wake_word + speech_to_text + text_to_speech nodes.

Topics:
  * publishes /user_speech_raw   (std_msgs/String)  — transcribed utterance
  * publishes /system_wake_state (std_msgs/Bool)    — latched; pulsed True after each utterance
  * subscribes /coaching_output  (std_msgs/String)  — spoken while gate is open
  * subscribes /tts_priority     (std_msgs/String)  — always spoken (motivation)

Params:
  ~audio_device          (default: "")     — mic device index or "" for system default
  ~wake_gate_seconds     (default: 30.0)   — how long coaching TTS gate stays open
  ~min_repeat_seconds    (default: 4.0)    — debounce identical TTS text
"""

# --- OLD: sounddevice-based recording — crashes with PortAudio ALSA assertion on Linux ---
# import sounddevice as sd
# import numpy as np
# def _query_native_rate(device):
#     info = sd.query_devices(device, "input")
#     return int(info["default_samplerate"])
# def _resample(audio, src_rate, dst_rate): ...
# class GroqVoiceIONode:
#     def _record_while_held(self):
#         with sd.InputStream(samplerate=self.native_rate, ...) as stream:
#             while self._space_held.is_set(): data, _ = stream.read(CHUNK) ...
# --- END OLD ---

# --- OLD: always-on listening (goon.py / Parakeet ASR + Kokoro TTS) ---
# import sys
# def _import_goon(): ...goon.load_asr(), goon.load_tts()...
# class VoiceIONode: ...listener.listen_once()...
# --- END OLD ---

import os
import signal
import subprocess
import tempfile
import threading
import time

import rospy
from groq import Groq
from pynput import keyboard
from std_msgs.msg import Bool, String

GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE = "autumn"

# arecord settings — 16 kHz mono S16_LE via PulseAudio (no PortAudio/ALSA assertion issues)
ARECORD_CMD    = ["arecord", "-D", "pulse", "-r", "16000", "-c", "1", "-f", "S16_LE"]


class GroqVoiceIONode:
    def __init__(self):
        self.client = Groq()

        self.wake_gate_secs = float(rospy.get_param("~wake_gate_seconds", 30.0))
        self.min_repeat     = float(rospy.get_param("~min_repeat_seconds", 4.0))

        self._gate_until  = 0.0
        self._last_spoken = {}
        self._stop        = threading.Event()
        self._space_held  = threading.Event()  # set while SPACE is pressed
        self._rec_proc    = None               # active arecord subprocess

        self.text_pub = rospy.Publisher("/user_speech_raw",   String, queue_size=4)
        self.wake_pub = rospy.Publisher("/system_wake_state", Bool,   queue_size=1, latch=True)
        self.wake_pub.publish(Bool(data=False))

        rospy.Subscriber("/coaching_output", String, self._on_coaching, queue_size=8)
        rospy.Subscriber("/tts_priority",    String, self._on_priority, queue_size=8)

        rospy.on_shutdown(self._shutdown)

        # Global keyboard listener for push-to-talk
        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._kb_listener.start()

        threading.Thread(target=self._ptt_loop, daemon=True).start()
        rospy.loginfo("voice_io (Groq PTT) ready — hold SPACE to talk (arecord/pulse)")

    # ------------------------------------------------------------------ #
    # Push-to-talk keyboard callbacks                                      #
    # ------------------------------------------------------------------ #

    def _on_key_press(self, key):
        if key == keyboard.Key.space and not self._space_held.is_set():
            rospy.loginfo("voice_io: SPACE held — recording...")
            self._space_held.set()

    def _on_key_release(self, key):
        if key == keyboard.Key.space:
            self._space_held.clear()

    # ------------------------------------------------------------------ #
    # PTT record loop                                                      #
    # ------------------------------------------------------------------ #

    def _ptt_loop(self):
        while not rospy.is_shutdown() and not self._stop.is_set():
            # Wait until SPACE is pressed
            if not self._space_held.wait(timeout=0.1):
                continue

            wav_bytes = self._record_while_held()

            if not wav_bytes or len(wav_bytes) < 100:
                rospy.loginfo("voice_io: no audio captured")
                continue

            rospy.loginfo("voice_io: transcribing %.1f kB of audio...", len(wav_bytes) / 1024)
            try:
                text = self._transcribe(wav_bytes)
            except Exception as exc:
                rospy.logwarn("voice_io STT error: %s", exc)
                continue

            if not text:
                rospy.loginfo("voice_io: empty transcription")
                continue

            rospy.loginfo("STT: %s", text)
            self._gate_until = time.time() + self.wake_gate_secs
            self.wake_pub.publish(Bool(data=True))
            self.text_pub.publish(String(data=text))
            threading.Timer(0.25, lambda: self.wake_pub.publish(Bool(data=False))).start()

    def _record_while_held(self) -> bytes:
        """Record via arecord/PulseAudio while SPACE is held. Returns raw WAV bytes."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        path = tmp.name

        cmd = ARECORD_CMD + [path]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._rec_proc = proc
            # Wait while SPACE is held
            while self._space_held.is_set() and not rospy.is_shutdown():
                rospy.sleep(0.05)
        finally:
            self._rec_proc = None
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.terminate()

        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception as exc:
            rospy.logwarn("voice_io: could not read wav file: %s", exc)
            return b""
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Groq STT                                                             #
    # ------------------------------------------------------------------ #

    def _transcribe(self, wav_bytes: bytes) -> str:
        """Send raw WAV bytes to Groq Whisper and return transcript."""
        result = self.client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=GROQ_STT_MODEL,
            temperature=0,
            response_format="text",
        )
        return (result or "").strip()

    # ------------------------------------------------------------------ #
    # Groq TTS                                                             #
    # ------------------------------------------------------------------ #

    def _speak(self, text: str):
        try:
            response = self.client.audio.speech.create(
                model=GROQ_TTS_MODEL,
                voice=GROQ_TTS_VOICE,
                response_format="wav",
                input=text,
            )
            wav_bytes = response.read()   # BinaryAPIResponse → bytes
            # Write to temp file and play via aplay (avoids PortAudio/ALSA issues)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(wav_bytes)
            tmp.close()
            subprocess.run(
                ["aplay", "-D", "pulse", tmp.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.unlink(tmp.name)
        except Exception as exc:
            rospy.logwarn("voice_io TTS failed: %s", exc)

    # ------------------------------------------------------------------ #
    # TTS callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _enqueue_tts(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        now = time.time()
        if now - self._last_spoken.get(text, 0.0) < self.min_repeat:
            return
        self._last_spoken[text] = now
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _on_coaching(self, msg: String):
        if time.time() < self._gate_until:
            self._enqueue_tts(msg.data)

    def _on_priority(self, msg: String):
        self._enqueue_tts(msg.data)

    def _shutdown(self):
        self._stop.set()
        self._space_held.clear()
        try:
            if self._rec_proc and self._rec_proc.poll() is None:
                self._rec_proc.terminate()
        except Exception:
            pass
        try:
            self._kb_listener.stop()
        except Exception:
            pass


# --- OLD VoiceIONode (Parakeet ASR + Kokoro TTS, always-on via goon.py) ---
# class VoiceIONode:
#     def __init__(self, goon):
#         ...load_asr(), load_tts(), VoiceListener, TTSPlayer...
#         ...always-on _listen_loop calling goon.beep + listener.listen_once()...
# def main():  # OLD
#     goon = _import_goon()
#     VoiceIONode(goon)
# --- END OLD ---


def main():
    rospy.init_node("voice_io_node")
    if not os.getenv("GROQ_API_KEY"):
        rospy.logfatal("voice_io: GROQ_API_KEY not set — node will not start")
        return
    GroqVoiceIONode()
    rospy.spin()


if __name__ == "__main__":
    main()
