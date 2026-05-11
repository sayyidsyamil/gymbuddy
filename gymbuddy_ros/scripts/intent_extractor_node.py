#!/usr/bin/env python3
"""Parses simple voice commands into structured IntentUpdate messages."""

import re

import rospy
from std_msgs.msg import String

from gymbuddy_ros.msg import IntentUpdate

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}


def parse_int(text: str):
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


class IntentExtractorNode:
    def __init__(self):
        self.pub = rospy.Publisher("/intent_update", IntentUpdate, queue_size=4)
        rospy.Subscriber("/user_speech_raw", String, self.on_speech, queue_size=4)
        rospy.loginfo("intent_extractor_node ready")

    def on_speech(self, msg: String):
        text = msg.data.lower().strip()
        if not text:
            return

        intent = self.classify(text)
        if intent is None:
            return

        out = IntentUpdate()
        out.header.stamp = rospy.Time.now()
        out.action = intent["action"]
        out.value = int(intent.get("value", 0))
        out.text = text
        self.pub.publish(out)
        rospy.loginfo("intent: %s value=%d", out.action, out.value)

    def classify(self, text: str):
        if re.search(r"\b(start|begin)\b.*\b(set|workout)\b", text) or text.startswith("start"):
            return {"action": "start_set"}
        if re.search(r"\b(stop|end|finish)\b.*\b(set|workout)\b", text) or text.startswith("stop"):
            return {"action": "stop_set"}
        if re.search(r"\b(reset|clear)\b", text):
            return {"action": "reset"}
        if re.search(r"\badd\b.*\breps?\b", text):
            value = parse_int(text)
            if value is not None:
                return {"action": "add_reps", "value": value}
        if re.search(r"\b(set|target).*\b(target|goal|reps?)\b", text):
            value = parse_int(text)
            if value is not None:
                return {"action": "update_target", "value": value}
        if re.search(r"\b(target|goal)\b", text):
            value = parse_int(text)
            if value is not None:
                return {"action": "update_target", "value": value}
        return None


def main():
    rospy.init_node("intent_extractor_node")
    IntentExtractorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
