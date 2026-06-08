#!/usr/bin/env python3
"""LLM decision node — coaching removed. Qwen GGUF is now used by intent_extractor_node."""

import rospy


def main():
    rospy.init_node("llm_decision_node")
    rospy.loginfo("llm_decision_node: coaching disabled — intent NLP handled by intent_extractor_node")
    rospy.spin()


if __name__ == "__main__":
    main()
