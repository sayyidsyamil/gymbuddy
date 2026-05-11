#!/usr/bin/env python3
"""LLM coach. Subscribes /user_speech_raw + /workout_stats and emits coaching text."""

import json
import os
import threading

import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import WorkoutStats

DEFAULT_PROMPT = (
    "You are GymBuddy, a quiet robotic fitness coach. Use only the measured JSON data. "
    "Do not invent injuries, weights, or hidden details. Give concise coaching: what went "
    "well, the main issue, and one next-set correction. Keep replies under 40 words.\n\n"
)


class LLMDecisionNode:
    def __init__(self):
        self.model_path = rospy.get_param("~model_path", os.getenv("GYMBUDDY_QWEN_GGUF", ""))
        self.n_ctx = int(rospy.get_param("~n_ctx", 4096))
        self.max_tokens = int(rospy.get_param("~max_tokens", 140))
        self.temperature = float(rospy.get_param("~temperature", 0.4))

        self.coach_pub = rospy.Publisher("/coaching_output", String, queue_size=4)

        self._llm = None
        self._llm_lock = threading.Lock()
        self._latest_stats = {}

        rospy.Subscriber("/workout_stats", WorkoutStats, self.on_stats, queue_size=4)
        rospy.Subscriber("/user_speech_raw", String, self.on_question, queue_size=4)

        if self.model_path and os.path.isfile(self.model_path):
            rospy.loginfo("llm_decision_node will load Qwen GGUF lazily from %s", self.model_path)
        else:
            rospy.logwarn("llm_decision_node has no GGUF model; falling back to rule-based coach")
        rospy.loginfo("llm_decision_node ready")

    def on_stats(self, msg: WorkoutStats):
        self._latest_stats = {
            "exercise": msg.exercise,
            "clean_reps": msg.clean_reps,
            "total_attempts": msg.total_attempts,
            "target_reps": msg.target_reps,
            "last_rep_seconds": round(msg.last_rep_seconds, 2),
            "last_min_angle": round(msg.last_min_angle, 1),
            "last_rep_issue": msg.last_rep_issue,
        }

    def on_question(self, msg: String):
        question = msg.data.strip()
        if not question:
            return
        threading.Thread(target=self._respond, args=(question,), daemon=True).start()

    def _respond(self, question: str):
        summary = dict(self._latest_stats)
        text = self._llm_reply(summary, question)
        if text:
            self.coach_pub.publish(String(data=text))

    def _llm_reply(self, summary: dict, question: str) -> str:
        if not self.model_path or not os.path.isfile(self.model_path):
            return self._fallback(summary, question)
        try:
            from llama_cpp import Llama
        except ImportError:
            return self._fallback(summary, question)

        with self._llm_lock:
            if self._llm is None:
                rospy.loginfo("loading Qwen GGUF (this is slow on first call)")
                self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)

        prompt = DEFAULT_PROMPT + f"Session JSON:\n{json.dumps(summary, indent=2)}\n\n"
        prompt += f"User question: {question}\nAnswer:"
        result = self._llm(prompt, max_tokens=self.max_tokens,
                           temperature=self.temperature, stop=["\n\n"])
        return result["choices"][0]["text"].strip()

    @staticmethod
    def _fallback(summary: dict, question: str) -> str:
        if not summary:
            return "I don't have any workout stats yet. Start a set first."
        clean = summary.get("clean_reps", 0)
        attempts = summary.get("total_attempts", 0)
        issue = summary.get("last_rep_issue") or "none"
        return (f"You have {clean} clean reps out of {attempts} attempts. "
                f"Last rep issue: {issue}. Question noted: {question[:60]}")


def main():
    rospy.init_node("llm_decision_node")
    LLMDecisionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
