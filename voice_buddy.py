#!/usr/bin/env /opt/homebrew/bin/python3.11
"""GymBuddy voice assistant. Wake word: 'buddy' or 'hey buddy'."""
import sys, warnings, queue, tempfile, os, urllib.request, json, threading, re, time
warnings.filterwarnings("ignore")

import sounddevice as sd
import soundfile as sf
import numpy as np
from mlx_audio.stt.utils import load_model as load_stt
from mlx_audio.tts.utils import load_model as load_tts

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 0.6
MIN_SPEECH = 0.4
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "gemma-4-e4b-uncensored-hauhaucs-aggressive"
SYSTEM_PROMPT = "You are GymBuddy, a friendly workout assistant. Reply in ONE short sentence."
MAX_TOKENS = 80
TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
VOICE = "af_heart"
SENTENCE_END = re.compile(r"[.!?\n]")

WAKE_PATTERNS = [
    re.compile(r"\bhey\s+buddy\b", re.IGNORECASE),
    re.compile(r"\bbuddy\b", re.IGNORECASE),
]
ACTIVE_TIMEOUT = 30.0

print("Loading STT model...")
stt = load_stt("mlx-community/parakeet-tdt-0.6b-v3")
print("Loading TTS model...")
tts = load_tts(TTS_MODEL)
TTS_SR = tts.sample_rate
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F2FF" "]+"
)
out_stream = sd.OutputStream(samplerate=TTS_SR, channels=1, dtype="float32")
out_stream.start()

def synth(text):
    text = EMOJI_RE.sub("", text).strip()
    if not text:
        return None
    chunks = []
    for r in tts.generate(text=text, voice=VOICE, speed=1.0, lang_code="en", stream=False):
        chunks.append(np.asarray(r.audio))
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)

print("Warming up TTS...")
synth("hi")
print("Ready. Say 'hey buddy' to wake me up. Ctrl+C to quit.\n")

audio_q = queue.Queue()
history = [{"role": "system", "content": SYSTEM_PROMPT}]
muted = threading.Event()

def callback(indata, frames, time_, status):
    if not muted.is_set():
        audio_q.put(indata.copy())

def transcribe(audio):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    result = stt.generate(tmp.name, verbose=False)
    os.unlink(tmp.name)
    return getattr(result, "text", "").strip()

def stream_llm(text, out_q):
    history.append({"role": "user", "content": text})
    body = json.dumps({
        "model": MODEL,
        "messages": history,
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }).encode()
    req = urllib.request.Request(LMSTUDIO_URL, data=body, headers={"Content-Type": "application/json"})
    full = []
    try:
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content", "")
                except Exception:
                    continue
                if delta:
                    full.append(delta)
                    out_q.put(delta)
    finally:
        out_q.put(None)
        history.append({"role": "assistant", "content": "".join(full)})

def play_worker(play_q):
    while True:
        audio = play_q.get()
        if audio is None:
            break
        try:
            out_stream.write(audio)
        except Exception as e:
            print(f"\n[audio error: {e}]")

def speak(text):
    muted.set()
    try:
        audio = synth(text)
        if audio is not None:
            out_stream.write(audio)
    finally:
        while not audio_q.empty():
            try: audio_q.get_nowait()
            except queue.Empty: break
        muted.clear()

def ask_and_speak(text):
    muted.set()
    try:
        token_q = queue.Queue()
        play_q = queue.Queue()
        threading.Thread(target=stream_llm, args=(text, token_q), daemon=True).start()
        player_t = threading.Thread(target=play_worker, args=(play_q,), daemon=True)
        player_t.start()

        buf = ""
        printed = 0
        while True:
            tok = token_q.get()
            if tok is None:
                tail = buf.strip()
                if tail:
                    print(buf[printed:], end="", flush=True)
                    audio = synth(tail)
                    if audio is not None:
                        play_q.put(audio)
                print()
                break
            buf += tok
            m = SENTENCE_END.search(buf)
            if m:
                end = m.end()
                sentence = buf[:end].strip()
                if sentence:
                    print(buf[printed:end], end="", flush=True)
                    printed = 0
                    audio = synth(sentence)
                    if audio is not None:
                        play_q.put(audio)
                buf = buf[end:]
        play_q.put(None)
        player_t.join()
    finally:
        while not audio_q.empty():
            try: audio_q.get_nowait()
            except queue.Empty: break
        muted.clear()

def strip_wake(text):
    for pat in WAKE_PATTERNS:
        text = pat.sub("", text, count=1)
    return re.sub(r"^[\s,.!?-]+", "", text).strip()

def has_wake(text):
    return any(pat.search(text) for pat in WAKE_PATTERNS)

active_until = 0.0

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        buffer = np.zeros((0, 1), dtype=np.float32)
        silence_samples = 0
        speech_samples = 0
        in_speech = False
        while True:
            chunk = audio_q.get()
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
                    text = transcribe(buffer)
                    if text:
                        is_active = time.time() < active_until
                        woke = has_wake(text)
                        if woke or is_active:
                            print(f"You: {text}")
                            query = strip_wake(text) if woke else text
                            if not query:
                                print("AI:  Yeah?")
                                speak("Yeah?")
                            else:
                                print("AI:  ", end="", flush=True)
                                try:
                                    ask_and_speak(query)
                                except Exception as e:
                                    print(f"\n[error: {e}]\n")
                            active_until = time.time() + ACTIVE_TIMEOUT
                        else:
                            print(f"(ignored: {text})")
                buffer = np.zeros((0, 1), dtype=np.float32)
                silence_samples = 0
                speech_samples = 0
                in_speech = False
except KeyboardInterrupt:
    print("\nBye.")
