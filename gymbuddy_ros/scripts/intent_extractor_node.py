#!/usr/bin/env python3
"""NLP intent extractor — Groq LLM primary, regex fallback.

Supported actions (published to /intent_update):
  start_set      — start a workout set
  stop_set       — stop / end the current set
  reset          — reset everything
  add_reps       — add N reps to the target
  remove_reps    — subtract N reps from the target
  update_target  — set target to exactly N reps
  none           — not a workout command
"""

import json
import os
import re
import threading

import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import IntentUpdate

GROQ_MODEL = "qwen/qwen3-32b"

GROQ_SYSTEM = (
    "/no_think\n"
    "You are a gym voice command parser. "
    "Reply ONLY with a single JSON object — no markdown, no explanation, no thinking.\n\n"
    "Available actions:\n"
    "  add_reps      — add N reps to target  (e.g. 'add 5', 'plus 3', 'tambah 5')\n"
    "  remove_reps   — reduce target by N    (e.g. 'minus 5', 'kurang 3', 'tolak 2')\n"
    "  update_target — set target to N       (e.g. 'set to 10', 'target 12', 'buat 8')\n"
    "  start_set     — begin a set           (e.g. 'start', 'go', 'mula')\n"
    "  stop_set      — end the set           (e.g. 'stop', 'done', 'habis', 'finish')\n"
    "  reset         — reset everything      (e.g. 'reset', 'mulakan semula')\n"
    "  none          — not a workout command\n\n"
    "Words like 'rep', 'reps', 'rap', 'wrap', 'wraps' all mean the same thing.\n"
    'Output exactly: {"action": "...", "value": <integer>}'
)

# STT often mishears "rep/reps" — normalise before parsing
_HOMOPHONE_MAP = [
    (r"\bwraps?\b",   "reps"),
    (r"\braps?\b",    "reps"),
    (r"\bwrap\b",     "rep"),
    (r"\brap\b",      "rep"),
    (r"\btargets?\b", "target"),
    (r"\bsets?\b",    "set"),
]


def _normalise(text: str) -> str:
    """Fix common STT homophones before intent classification."""
    import re as _re
    t = text
    for pattern, replacement in _HOMOPHONE_MAP:
        t = _re.sub(pattern, replacement, t, flags=_re.IGNORECASE)
    return t

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty five": 25, "thirty": 30,
    "forty": 40, "fifty": 50,
}


def _extract_number(text: str):
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    tl = text.lower()
    for word, val in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", tl):
            return val
    return None


def _regex_classify(text: str):
    t = _normalise(text).lower()
    v = _extract_number(t)

    if re.search(r"\b(start|begin|let'?s go|go ahead|mula|mulakan)\b", t):
        return "start_set", 0
    if re.search(r"\b(stop|end|finish|done|pause|habis|berhenti)\b", t):
        return "stop_set", 0
    if re.search(r"\b(reset|clear|start over|mulakan semula)\b", t):
        return "reset", 0
    if re.search(r"\b(remove|subtract|minus|take off|reduce|less|kurang|tolak)\b", t):
        if v is not None:
            return "remove_reps", v
    if re.search(r"\b(add|tambah|plus)\b", t):
        if v is not None:
            return "add_reps", v
    # "set X to N" / "set to N" / "target N" / "buat N" — any number after a set/target word
    if re.search(r"\b(set|target|goal|buat|jadikan|change|make)\b", t):
        if v is not None:
            return "update_target", v
    # bare number with "rep/reps" nearby → treat as set target
    if v is not None and re.search(r"\breps?\b", t):
        return "update_target", v
    return None, 0


class IntentExtractorNode:
    def __init__(self):
        self._groq_client = None
        self._groq_lock   = threading.Lock()

        self.pub = rospy.Publisher("/intent_update", IntentUpdate, queue_size=4)
        rospy.Subscriber("/user_speech_raw", String, self.on_speech, queue_size=4)

        if os.getenv("GROQ_API_KEY"):
            threading.Thread(target=self._init_groq, daemon=True).start()
            rospy.loginfo("intent_extractor: Groq LLM active (regex until client ready)")
        else:
            rospy.logwarn("intent_extractor: no GROQ_API_KEY — regex only")

    def _init_groq(self):
        try:
            from groq import Groq
            with self._groq_lock:
                self._groq_client = Groq()
            rospy.loginfo("intent_extractor: Groq client ready")
        except Exception as e:
            rospy.logwarn("intent_extractor: Groq init failed (%s) — regex only", e)

    # ------------------------------------------------------------------ #
    # Speech callback                                                      #
    # ------------------------------------------------------------------ #

    def on_speech(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        threading.Thread(target=self._classify_and_publish,
                         args=(text,), daemon=True).start()

    def _classify_and_publish(self, text: str):
        text = _normalise(text)   # fix homophones before any classification
        action, value = self._classify(text)
        if action is None or action == "none":
            rospy.loginfo("intent_extractor: no actionable intent in: '%s'", text)
            return
        out = IntentUpdate()
        out.header.stamp = rospy.Time.now()
        out.action = action
        out.value  = int(value)
        out.text   = text
        self.pub.publish(out)
        rospy.loginfo("intent_extractor: %s value=%d  (from: '%s')", action, value, text)

    # ------------------------------------------------------------------ #
    # Classification — Groq first, regex fallback                         #
    # ------------------------------------------------------------------ #

    def _classify(self, text: str):
        with self._groq_lock:
            client = self._groq_client

        if client is not None:
            try:
                return self._groq_classify(client, text)
            except Exception as e:
                rospy.logwarn("intent_extractor: Groq failed (%s) — regex fallback", e)

        return _regex_classify(text)

    def _groq_classify(self, client, text: str):
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": GROQ_SYSTEM},
                {"role": "user",   "content": f'Command: "{text}"'},
            ],
            max_completion_tokens=60,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = raw.strip("`").strip()

        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not m:
            rospy.logwarn("intent_extractor: Groq non-JSON '%s' — regex fallback", raw)
            return _regex_classify(text)

        try:
            data   = json.loads(m.group(0))
            action = data.get("action", "none")
            value  = int(data.get("value", 0))
            rospy.logdebug("intent Groq: '%s' → %s(%d)", text, action, value)
            return action, value
        except (json.JSONDecodeError, ValueError) as e:
            rospy.logwarn("intent_extractor: JSON parse error (%s) — regex fallback", e)
            return _regex_classify(text)


def main():
    rospy.init_node("intent_extractor_node")
    IntentExtractorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
