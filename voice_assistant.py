"""Voice assistant for GymBuddy. Wake word + voice-to-voice using the loaded GGUF."""
import warnings, queue, tempfile, os, threading, re, time
warnings.filterwarnings("ignore")

import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 0.6
MIN_SPEECH = 0.4
SYSTEM_PROMPT = (
    "You are GymBuddy, a friendly voice workout coach. Reply in ONE short sentence. "
    "Be conversational. Use the live workout stats only if the user asks about them."
)
MAX_TOKENS = 80
VOICE = "af_heart"
TTS_MODEL_NAME = "mlx-community/Kokoro-82M-bf16"
STT_MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"

WAKE_PATTERNS = [
    re.compile(r"\bhey\s+buddy\b", re.IGNORECASE),
    re.compile(r"\bbuddy\b", re.IGNORECASE),
]
ACTIVE_TIMEOUT = 30.0
SENTENCE_END = re.compile(r"[.!?\n]")
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F2FF" "]+"
)


def _strip_wake(text):
    for pat in WAKE_PATTERNS:
        text = pat.sub("", text, count=1)
    return re.sub(r"^[\s,.!?-]+", "", text).strip()


def _has_wake(text):
    return any(pat.search(text) for pat in WAKE_PATTERNS)


class VoiceAssistant:
    def __init__(self, llm_call, get_context=None):
        """
        llm_call: callable(prompt: str) -> Iterator[str]  (yields token deltas)
        get_context: optional callable() -> str  (extra context appended to system prompt)
        """
        self.llm_call = llm_call
        self.get_context = get_context or (lambda: "")
        self.stop_event = threading.Event()
        self.audio_q = queue.Queue()
        self.muted = threading.Event()
        self._stt = None
        self._tts = None
        self._tts_sr = 24000
        self._out_stream = None
        self.active_until = 0.0

    def _load(self):
        from mlx_audio.stt.utils import load_model as load_stt
        from mlx_audio.tts.utils import load_model as load_tts
        print("[voice] loading STT...")
        self._stt = load_stt(STT_MODEL_NAME)
        print("[voice] loading TTS...")
        self._tts = load_tts(TTS_MODEL_NAME)
        self._tts_sr = self._tts.sample_rate
        self._out_stream = sd.OutputStream(samplerate=self._tts_sr, channels=1, dtype="float32")
        self._out_stream.start()
        print("[voice] warming TTS...")
        self._synth("hi")
        print("[voice] ready. Say 'hey buddy' to wake me.")

    def _synth(self, text):
        text = EMOJI_RE.sub("", text).strip()
        if not text:
            return None
        chunks = []
        for r in self._tts.generate(text=text, voice=VOICE, speed=1.0, lang_code="en", stream=False):
            chunks.append(np.asarray(r.audio))
        if not chunks:
            return None
        return np.concatenate(chunks).astype(np.float32)

    def _transcribe(self, audio):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, SAMPLE_RATE)
        try:
            result = self._stt.generate(tmp.name, verbose=False)
        finally:
            os.unlink(tmp.name)
        return getattr(result, "text", "").strip()

    def _play(self, audio):
        if audio is None:
            return
        try:
            self._out_stream.write(audio)
        except Exception as e:
            print(f"[voice] audio error: {e}")

    def _speak(self, text):
        self._play(self._synth(text))

    def _audio_callback(self, indata, frames, time_, status):
        if not self.muted.is_set():
            self.audio_q.put(indata.copy())

    def _build_prompt(self, user_text):
        ctx = self.get_context()
        sys = SYSTEM_PROMPT + (f"\n\nLive workout stats:\n{ctx}" if ctx else "")
        return f"<start_of_turn>system\n{sys}<end_of_turn>\n<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

    def _ask_and_speak(self, user_text):
        self.muted.set()
        try:
            prompt = self._build_prompt(user_text)
            play_q = queue.Queue()
            done = threading.Event()

            def player():
                while not done.is_set() or not play_q.empty():
                    try:
                        audio = play_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    self._play(audio)

            player_t = threading.Thread(target=player, daemon=True)
            player_t.start()

            buf = ""
            try:
                for delta in self.llm_call(prompt):
                    buf += delta
                    print(delta, end="", flush=True)
                    m = SENTENCE_END.search(buf)
                    if m:
                        end = m.end()
                        sentence = buf[:end].strip()
                        if sentence:
                            audio = self._synth(sentence)
                            if audio is not None:
                                play_q.put(audio)
                        buf = buf[end:]
                tail = buf.strip()
                if tail:
                    audio = self._synth(tail)
                    if audio is not None:
                        play_q.put(audio)
            finally:
                print()
                done.set()
                player_t.join()
        finally:
            while not self.audio_q.empty():
                try: self.audio_q.get_nowait()
                except queue.Empty: break
            self.muted.clear()

    def _loop(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=self._audio_callback):
            buffer = np.zeros((0, 1), dtype=np.float32)
            silence_samples = 0
            speech_samples = 0
            in_speech = False
            while not self.stop_event.is_set():
                try:
                    chunk = self.audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                buffer = np.concatenate([buffer, chunk])
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms > SILENCE_THRESHOLD:
                    in_speech = True
                    speech_samples += len(chunk)
                    silence_samples = 0
                elif in_speech:
                    silence_samples += len(chunk)
                if in_speech and silence_samples > SILENCE_DURATION * SAMPLE_RATE:
                    if speech_samples > MIN_SPEECH * SAMPLE_RATE:
                        try:
                            text = self._transcribe(buffer)
                        except Exception as e:
                            print(f"[voice] STT error: {e}")
                            text = ""
                        if text:
                            is_active = time.time() < self.active_until
                            woke = _has_wake(text)
                            if woke or is_active:
                                print(f"\nYou: {text}")
                                query = _strip_wake(text) if woke else text
                                if not query:
                                    print("AI:  Yeah?")
                                    self._speak("Yeah?")
                                else:
                                    print("AI:  ", end="", flush=True)
                                    try:
                                        self._ask_and_speak(query)
                                    except Exception as e:
                                        print(f"\n[voice] error: {e}")
                                self.active_until = time.time() + ACTIVE_TIMEOUT
                    buffer = np.zeros((0, 1), dtype=np.float32)
                    silence_samples = 0
                    speech_samples = 0
                    in_speech = False

    def start(self):
        self._load()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
