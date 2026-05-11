import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_NAME = "yolo26n-pose.pt"
POSE_CONFIDENCE = 0.35
KEYPOINT_CONFIDENCE = 0.35

EXTENDED_ANGLE = 155
CURLING_ANGLE = 135
TOP_ANGLE = 70
PARTIAL_TOP_ANGLE = 100
FAST_REP_SECONDS = 1.0

LEFT_ARM = {"name": "left", "shoulder": 5, "elbow": 7, "wrist": 9}
RIGHT_ARM = {"name": "right", "shoulder": 6, "elbow": 8, "wrist": 10}
QWEN_MODEL = None


@dataclass
class ArmPose:
    side: str
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    confidence: float
    angle: float
    upper_arm_length: float


@dataclass
class RepAttempt:
    number: int
    clean: bool
    duration_seconds: float
    min_angle: float
    max_elbow_drift: float
    issues: List[str] = field(default_factory=list)


@dataclass
class CurlSession:
    active: bool = False
    started_at: Optional[float] = None
    clean_reps: int = 0
    total_attempts: int = 0
    reps: List[RepAttempt] = field(default_factory=list)
    state: str = "ready"
    cue: str = "Press Space to start"
    rep_started_at: Optional[float] = None
    rep_start_elbow: Optional[np.ndarray] = None
    min_angle: float = 180.0
    max_elbow_drift: float = 0.0
    saw_top: bool = False
    saw_partial: bool = False
    perception_warnings: Counter = field(default_factory=Counter)

    def start(self):
        self.active = True
        self.started_at = time.time()
        self.clean_reps = 0
        self.total_attempts = 0
        self.reps.clear()
        self.state = "ready"
        self.cue = "Find full extension"
        self._reset_rep_trackers()

    def stop(self):
        self.active = False
        self.state = "review"
        self.cue = "Set complete"

    def _reset_rep_trackers(self):
        self.rep_started_at = None
        self.rep_start_elbow = None
        self.min_angle = 180.0
        self.max_elbow_drift = 0.0
        self.saw_top = False
        self.saw_partial = False

    def update(self, arm: Optional[ArmPose]):
        if not self.active:
            return

        if arm is None:
            self.cue = "Arm not visible"
            self.perception_warnings["arm_not_visible"] += 1
            return

        if arm.confidence < KEYPOINT_CONFIDENCE:
            self.cue = "Low confidence"
            self.perception_warnings["low_confidence"] += 1
            return

        angle = arm.angle
        self.min_angle = min(self.min_angle, angle)

        if self.rep_start_elbow is not None:
            drift = float(np.linalg.norm(arm.elbow - self.rep_start_elbow))
            self.max_elbow_drift = max(self.max_elbow_drift, drift)

        if self.state == "ready":
            if angle >= EXTENDED_ANGLE:
                self.cue = "Ready"
            elif angle < CURLING_ANGLE:
                self._begin_rep(arm)
                self.state = "curling"
                self.cue = "Curl higher"

        elif self.state == "curling":
            if angle <= TOP_ANGLE:
                self.saw_top = True
                self.state = "top"
                self.cue = "Control down"
            elif angle <= PARTIAL_TOP_ANGLE:
                self.saw_partial = True
                self.cue = "Almost there"
            else:
                self.cue = "Curl higher"

        elif self.state == "top":
            if angle > TOP_ANGLE + 15:
                self.state = "lowering"
                self.cue = "Full extension"

        elif self.state == "lowering":
            if angle >= EXTENDED_ANGLE:
                self._finish_rep(arm, finished_with_full_extension=True)
            elif angle < CURLING_ANGLE - 10:
                self._finish_rep(arm, finished_with_full_extension=False)
            elif angle >= CURLING_ANGLE:
                self.cue = "Extend lower"

        if self.state == "curling" and self.rep_started_at is not None:
            if angle >= EXTENDED_ANGLE and (self.saw_partial or self.min_angle < CURLING_ANGLE):
                self._finish_rep(arm, finished_with_full_extension=True)

    def _begin_rep(self, arm: ArmPose):
        self.rep_started_at = time.time()
        self.rep_start_elbow = arm.elbow.copy()
        self.min_angle = arm.angle
        self.max_elbow_drift = 0.0
        self.saw_top = False
        self.saw_partial = False

    def _finish_rep(self, arm: ArmPose, finished_with_full_extension: bool):
        if self.rep_started_at is None:
            self._reset_rep_trackers()
            self.state = "ready"
            self.cue = "Ready"
            return

        duration = time.time() - self.rep_started_at
        drift_threshold = max(45.0, arm.upper_arm_length * 0.45)
        issues = []

        if not self.saw_top:
            issues.append("partial_curl")
        if not finished_with_full_extension:
            issues.append("incomplete_extension")
        if duration < FAST_REP_SECONDS:
            issues.append("too_fast")
        if self.max_elbow_drift > drift_threshold:
            issues.append("elbow_drift")

        clean = not issues
        self.total_attempts += 1
        if clean:
            self.clean_reps += 1
            self.cue = "Rep counted"
        else:
            self.cue = cue_for_issues(issues)

        self.reps.append(
            RepAttempt(
                number=self.total_attempts,
                clean=clean,
                duration_seconds=round(duration, 2),
                min_angle=round(self.min_angle, 1),
                max_elbow_drift=round(self.max_elbow_drift, 1),
                issues=issues,
            )
        )
        self._reset_rep_trackers()
        self.state = "ready"

    def summary(self):
        issue_counts = Counter(issue for rep in self.reps for issue in rep.issues)
        durations = [rep.duration_seconds for rep in self.reps if rep.duration_seconds > 0]
        avg_tempo = round(sum(durations) / len(durations), 2) if durations else 0

        return {
            "exercise": "one_arm_bicep_curl",
            "clean_reps": self.clean_reps,
            "total_attempts": self.total_attempts,
            "issues": dict(issue_counts),
            "perception_warnings": dict(self.perception_warnings),
            "average_rep_seconds": avg_tempo,
            "reps": [
                {
                    "number": rep.number,
                    "clean": rep.clean,
                    "duration_seconds": rep.duration_seconds,
                    "min_angle": rep.min_angle,
                    "issues": rep.issues,
                }
                for rep in self.reps
            ],
        }


def cue_for_issues(issues):
    if "partial_curl" in issues:
        return "Curl higher"
    if "incomplete_extension" in issues:
        return "Extend lower"
    if "elbow_drift" in issues:
        return "Keep elbow still"
    if "too_fast" in issues:
        return "Slow down"
    return "Try again"


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 180.0
    cosine = np.dot(ba, bc) / denominator
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def select_arm(result):
    if result.keypoints is None or result.keypoints.xy is None:
        return None
    if len(result.keypoints.xy) == 0:
        return None

    xy = result.keypoints.xy[0].cpu().numpy()
    conf = result.keypoints.conf[0].cpu().numpy()

    candidates = []
    for arm_def in (LEFT_ARM, RIGHT_ARM):
        indices = [arm_def["shoulder"], arm_def["elbow"], arm_def["wrist"]]
        arm_conf = float(np.mean(conf[indices]))
        if arm_conf <= 0:
            continue

        shoulder = xy[arm_def["shoulder"]]
        elbow = xy[arm_def["elbow"]]
        wrist = xy[arm_def["wrist"]]
        angle = angle_between(shoulder, elbow, wrist)
        upper_arm_length = float(np.linalg.norm(shoulder - elbow))

        candidates.append(
            ArmPose(
                side=arm_def["name"],
                shoulder=shoulder,
                elbow=elbow,
                wrist=wrist,
                confidence=arm_conf,
                angle=angle,
                upper_arm_length=upper_arm_length,
            )
        )

    if not candidates:
        return None

    return max(candidates, key=lambda arm: arm.confidence)


def draw_arm(frame, arm):
    if arm is None:
        return

    points = [arm.shoulder, arm.elbow, arm.wrist]
    for start, end in zip(points, points[1:]):
        cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (0, 220, 255), 4)
    for point in points:
        cv2.circle(frame, tuple(point.astype(int)), 7, (0, 255, 120), -1)


def draw_panel(frame, session, arm, fps, debug):
    height, width = frame.shape[:2]
    panel_w = 390
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, height), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    status = "LIVE SET" if session.active else "READY"
    if session.state == "review":
        status = "SET REVIEW"

    cv2.putText(frame, "GYMBUDDY", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "LOCAL AI CURL COACH", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 210, 255), 1)
    cv2.putText(frame, status, (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (190, 190, 190), 2)

    cv2.putText(frame, str(session.clean_reps).zfill(2), (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 255), 7)
    cv2.putText(frame, "CLEAN REPS", (36, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 2)

    cue_color = (0, 255, 120) if session.cue in ("Ready", "Rep counted", "Set complete") else (0, 220, 255)
    cv2.putText(frame, session.cue.upper(), (30, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.72, cue_color, 2)

    help_lines = [
        "Space: start/stop",
        "D: debug",
        "R: reset",
        "Q/Esc: quit",
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(frame, line, (30, height - 108 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 205, 205), 1)

    if debug:
        angle = "--" if arm is None else f"{arm.angle:.1f}"
        conf = "--" if arm is None else f"{arm.confidence:.2f}"
        side = "--" if arm is None else arm.side
        debug_lines = [
            f"state: {session.state}",
            f"arm: {side}",
            f"elbow angle: {angle}",
            f"confidence: {conf}",
            f"attempts: {session.total_attempts}",
            f"fps: {fps:.1f}",
        ]
        for i, line in enumerate(debug_lines):
            cv2.putText(frame, line, (width - 300, 34 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)


def fallback_coach(summary, question=None):
    issues = summary["issues"]
    if summary["total_attempts"] == 0:
        return "I did not count any curl attempts. Make sure your shoulder, elbow, and wrist are visible from the side."

    if not issues:
        main_issue = "No major form issue detected."
        correction = "Keep the same controlled tempo next set."
    else:
        issue, _ = max(issues.items(), key=lambda item: item[1])
        main_issue = issue.replace("_", " ")
        corrections = {
            "partial_curl": "Curl a little higher before lowering.",
            "incomplete_extension": "Reach full extension at the bottom before starting the next rep.",
            "elbow_drift": "Keep your elbow pinned near the same spot instead of swinging forward.",
            "too_fast": "Slow the curl down so each rep stays controlled.",
        }
        correction = corrections.get(issue, "Focus on cleaner, slower reps next set.")

    if question:
        return f"Based on this set, the main thing to know is: {main_issue}. {correction}"

    return (
        f"You completed {summary['clean_reps']} clean reps out of {summary['total_attempts']} attempts. "
        f"Main issue: {main_issue}. Next set: {correction}"
    )


def qwen_coach(summary, question=None):
    global QWEN_MODEL
    model_path = os.getenv("GYMBUDDY_QWEN_GGUF")
    if not model_path:
        return fallback_coach(summary, question)

    try:
        from llama_cpp import Llama
    except ImportError:
        return fallback_coach(summary, question)

    prompt = (
        "You are GymBuddy, a quiet robotic fitness coach. Use only the measured JSON data. "
        "Do not invent injuries, weights, or hidden details. Give concise coaching: what went well, "
        "main issue, and one next-set correction.\n\n"
        f"Session JSON:\n{json.dumps(summary, indent=2)}\n\n"
    )
    if question:
        prompt += f"User question: {question}\nAnswer:"
    else:
        prompt += "Coach summary:"

    if QWEN_MODEL is None:
        QWEN_MODEL = Llama(model_path=model_path, n_ctx=4096, verbose=False)

    result = QWEN_MODEL(prompt, max_tokens=140, temperature=0.4, stop=["\n\n"])
    return result["choices"][0]["text"].strip()


def print_set_review(session):
    summary = session.summary()
    print("\n=== GymBuddy Set Review ===")
    print(json.dumps(summary, indent=2))
    print("\nCoach:")
    print(qwen_coach(summary))

    while True:
        question = input("\nAsk coach a follow-up, or press Enter to finish: ").strip()
        if not question:
            break
        print(qwen_coach(summary, question))


def main():
    model = YOLO(MODEL_NAME)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not camera.isOpened():
        print("Could not open camera. Check camera permissions and try again.")
        return

    session = CurlSession()
    debug = True
    last_time = time.time()
    fps = 0.0

    print("GymBuddy Curl Coach started.")
    print("Press Space to start/stop a set. Press D for debug, R to reset, Q/Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read from camera.")
                break

            result = model.predict(frame, conf=POSE_CONFIDENCE, verbose=False)[0]
            arm = select_arm(result)
            session.update(arm)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 0.001))
            last_time = now

            annotated_frame = frame.copy()
            draw_arm(annotated_frame, arm)
            draw_panel(annotated_frame, session, arm, fps, debug)

            cv2.imshow("GymBuddy Curl Coach", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("r"):
                session = CurlSession()
            if key == 32:
                if session.active:
                    session.stop()
                    print_set_review(session)
                else:
                    session.start()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
