"""Voice assistant for GymBuddy — Groq cloud STT + LLM + TTS."""
import warnings, queue, tempfile, os, threading, re, time, io
warnings.filterwarnings("ignore")

import sounddevice as sd
import soundfile as sf
import numpy as np
from groq import Groq

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 0.6
MIN_SPEECH = 0.4

GROQ_STT_MODEL   = "whisper-large-v3-turbo"
GROQ_LLM_MODEL   = "qwen/qwen3-32b"
GROQ_TTS_MODEL   = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE   = "autumn"

SYSTEM_PROMPT = (
    "You are GymBuddy, a friendly voice workout coach. "
    "Reply in ONE short sentence. Be conversational. "
    "Use the live workout stats only if the user asks about them. "
    "Do not include hidden reasoning or <think> tags. /no_think"
)

WAKE_PATTERNS = [
    re.compile(r"\bhey\s+buddy\b", re.IGNORECASE),
    re.compile(r"\bbuddy\b",       re.IGNORECASE),
]
ACTIVE_TIMEOUT = 30.0
SENTENCE_END   = re.compile(r"[.!?\n]")
EMOJI_RE       = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F2FF" "]+"
)

_CMD_PATTERNS = [
    (re.compile(r"\b(?:add|at)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:more\s+)?(?:r[ae]ps?)?\b", re.IGNORECASE),   "add_reps"),
    (re.compile(r"\bset\s+(?:(?:the\s+)?target\s+(?:to\s+)?|it\s+to\s+)(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*r[ae]ps?\b", re.IGNORECASE), "set_target"),
    (re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+more(?:\s+r[ae]ps?)?(?:\s+\w+)?\b", re.IGNORECASE), "add_reps"),
    (re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+r[ae]ps?\s*(?:please|now)?\b", re.IGNORECASE), "set_target"),
    (re.compile(r"\b(?:clear|remove|cancel)\s+(?:the\s+)?target\b", re.IGNORECASE), "clear_target"),
    (re.compile(r"\b(?:start|begin)\s+(?:the\s+)?(?:set|workout|session)\b", re.IGNORECASE), "start"),
    (re.compile(r"\b(?:stop|finish|end|done)\s*(?:the\s+)?(?:set|workout|session)?\b", re.IGNORECASE), "stop"),
    (re.compile(r"\breset\b", re.IGNORECASE), "reset"),
]

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _strip_wake(text):
    for pat in WAKE_PATTERNS:
        text = pat.sub("", text, count=1)
    return re.sub(r"^[\s,.!?-]+", "", text).strip()


def _has_wake(text):
    return any(pat.search(text) for pat in WAKE_PATTERNS)


def _parse_command(text):
    for pattern, cmd in _CMD_PATTERNS:
        m = pattern.search(text)
        if m:
            raw_value = m.group(1).lower() if m.lastindex else ""
            value = int(raw_value) if raw_value.isdigit() else NUMBER_WORDS.get(raw_value, 0)
            return cmd, value
    return None


class VoiceAssistant:
    def __init__(self, get_context=None, on_command=None):
        """
        get_context: optional callable() -> str  (returns live workout stats)
        on_command:  optional callable(cmd, value) -> str  (mutates session, returns reply)
        """
        self.get_context = get_context or (lambda: "")
        self.on_command  = on_command
        self.client      = Groq()
        self.stop_event  = threading.Event()
        self.audio_q     = queue.Queue()
        self.muted       = threading.Event()
        self.active_until = 0.0

    # ── audio helpers ──────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_, status):
        if not self.muted.is_set():
            self.audio_q.put(indata.copy())

    def _transcribe(self, audio):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, SAMPLE_RATE)
        try:
            with open(tmp.name, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    file=(tmp.name, f.read()),
                    model=GROQ_STT_MODEL,
                    temperature=0,
                    response_format="verbose_json",
                )
        finally:
            os.unlink(tmp.name)
        return (result.text or "").strip()

    def _synth(self, text):
        text = EMOJI_RE.sub("", text).strip()
        if not text:
            return None, None
        try:
            resp = self.client.audio.speech.create(
                model=GROQ_TTS_MODEL,
                voice=GROQ_TTS_VOICE,
                response_format="wav",
                input=text,
            )
            if hasattr(resp, "read"):
                content = resp.read()
            else:
                content = getattr(resp, "content", resp)
            buf = io.BytesIO(content)
            audio, sr = sf.read(buf, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio, sr
        except Exception as e:
            print(f"\n[voice] TTS error: {e}")
            return None, None

    def _play(self, audio, sr):
        if audio is None:
            return
        try:
            sd.play(audio, sr)
            sd.wait()
        except Exception as e:
            print(f"\n[voice] playback error: {e}")

    def _speak(self, text):
        audio, sr = self._synth(text)
        self._play(audio, sr)

    # ── LLM streaming ──────────────────────────────────────────────────────

    def _build_messages(self, user_text):
        ctx = self.get_context()
        sys = SYSTEM_PROMPT + (f"\n\nLive workout stats:\n{ctx}" if ctx else "")
        return [
            {"role": "system",  "content": sys},
            {"role": "user",    "content": user_text},
        ]

    def _ask_and_speak(self, user_text):
        self.muted.set()
        try:
            play_q = queue.Queue()
            done   = threading.Event()

            def player():
                while not done.is_set() or not play_q.empty():
                    try:
                        audio, sr = play_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    self._play(audio, sr)

            player_t = threading.Thread(target=player, daemon=True)
            player_t.start()

            buf     = ""
            full    = []
            stream  = self.client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=self._build_messages(user_text),
                temperature=0.6,
                max_completion_tokens=80,
                stream=True,
            )
            try:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    buf  += delta
                    full.append(delta)
                    print(delta, end="", flush=True)
                    m = SENTENCE_END.search(buf)
                    if m:
                        end      = m.end()
                        sentence = buf[:end].strip()
                        if sentence:
                            audio, sr = self._synth(sentence)
                            if audio is not None:
                                play_q.put((audio, sr))
                        buf = buf[end:]
                tail = buf.strip()
                if tail:
                    audio, sr = self._synth(tail)
                    if audio is not None:
                        play_q.put((audio, sr))
            finally:
                print()
                done.set()
                player_t.join()
        finally:
            while not self.audio_q.empty():
                try: self.audio_q.get_nowait()
                except queue.Empty: break
            self.muted.clear()

    # ── main loop ──────────────────────────────────────────────────────────

    def _loop(self):
        print("[voice] ready. Say 'hey buddy' to wake me.")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=self._audio_callback):
            buffer         = np.zeros((0, 1), dtype=np.float32)
            silence_samples = 0
            speech_samples  = 0
            in_speech       = False
            while not self.stop_event.is_set():
                try:
                    chunk = self.audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                buffer = np.concatenate([buffer, chunk])
                rms    = float(np.sqrt(np.mean(chunk**2)))
                if rms > SILENCE_THRESHOLD:
                    in_speech       = True
                    speech_samples += len(chunk)
                    silence_samples = 0
                elif in_speech:
                    silence_samples += len(chunk)
                if in_speech and silence_samples > SILENCE_DURATION * SAMPLE_RATE:
                    if speech_samples > MIN_SPEECH * SAMPLE_RATE:
                        try:
                            text = self._transcribe(buffer)
                        except Exception as e:
                            print(f"\n[voice] STT error: {e}")
                            text = ""
                        if text:
                            is_active = time.time() < self.active_until
                            woke      = _has_wake(text)
                            if woke or is_active:
                                print(f"\nYou: {text}")
                                query = _strip_wake(text) if woke else text
                                if not query:
                                    print("AI:  Yeah?")
                                    self._speak("Yeah?")
                                else:
                                    parsed = _parse_command(query) if self.on_command else None
                                    if parsed:
                                        cmd, value = parsed
                                        try:
                                            reply = self.on_command(cmd, value)
                                        except Exception as e:
                                            reply = None
                                            print(f"\n[voice] command error: {e}")
                                        if reply:
                                            print(f"AI:  {reply}")
                                            self._speak(reply)
                                    else:
                                        print("AI:  ", end="", flush=True)
                                        try:
                                            self._ask_and_speak(query)
                                        except Exception as e:
                                            print(f"\n[voice] error: {e}")
                                self.active_until = time.time() + ACTIVE_TIMEOUT
                    buffer          = np.zeros((0, 1), dtype=np.float32)
                    silence_samples = 0
                    speech_samples  = 0
                    in_speech       = False

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
