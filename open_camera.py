import json
import math
import os
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_NAME = "yolo26n-pose.pt"
APP_VERSION = "GymBuddy standalone 2026-05-17 camera+tts-fix"
POSE_CONFIDENCE = 0.35
KEYPOINT_CONFIDENCE = 0.35

EXTENDED_ANGLE = 155
CURLING_ANGLE = 135
TOP_ANGLE = 90
PARTIAL_TOP_ANGLE = 120
FAST_REP_SECONDS = 0.6

LEFT_ARM = {"name": "left", "shoulder": 5, "elbow": 7, "wrist": 9}
RIGHT_ARM = {"name": "right", "shoulder": 6, "elbow": 8, "wrist": 10}
QWEN_MODEL = None
QWEN_LOCK = __import__("threading").Lock()
COACH_EXECUTOR = ThreadPoolExecutor(max_workers=1)
GROQ_CLIENT = None
GROQ_LLM_MODEL = "qwen/qwen3-32b"
CAMERA_RETRY_SECONDS = 3.0
CAMERA_MAX_READ_FAILURES = 30
CAMERA_WARMUP_FRAMES = 12


def groq_client():
    global GROQ_CLIENT
    if GROQ_CLIENT is None:
        from groq import Groq
        GROQ_CLIENT = Groq()
    return GROQ_CLIENT


def coach_model_path():
    return os.getenv("GYMBUDDY_GGUF") or os.getenv("GYMBUDDY_QWEN_GGUF")


def camera_indexes():
    raw_value = os.getenv("GYMBUDDY_CAMERA_INDEX", "auto").strip().lower()
    if raw_value in ("", "auto"):
        return [0, 1, 2]
    try:
        return [int(raw_value)]
    except ValueError:
        print(f"Invalid GYMBUDDY_CAMERA_INDEX={raw_value!r}; using auto camera scan.")
        return [0, 1, 2]


def _read_warmup_frame(camera):
    for _ in range(CAMERA_WARMUP_FRAMES):
        ok, frame = camera.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.05)
    return None


def open_camera_device(index):
    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
    backends.append(("default", 0))

    for name, backend in backends:
        camera = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        if camera.isOpened():
            first_frame = _read_warmup_frame(camera)
            if first_frame is None:
                print(f"Camera {index} opened with {name}, but did not return frames.")
                camera.release()
                continue
            print(f"Camera {index} opened with {name}.")
            return camera
        camera.release()
    return None


def camera_help(indexes):
    index_list = ", ".join(str(index) for index in indexes)
    return (
        f"Could not read from camera index(es): {index_list}. On macOS, enable Camera permission "
        "for Terminal/Codex/Python in System Settings > Privacy & Security > Camera. "
        "If OpenCV says out of bound, that camera index does not exist on this Mac. "
        "If you pressed Ctrl+Z earlier, run: pkill -f open_camera.py"
    )


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
    coach_tip: str = "Start a set for live coaching"
    last_coached_attempt: int = 0
    rep_started_at: Optional[float] = None
    rep_start_elbow: Optional[np.ndarray] = None
    min_angle: float = 180.0
    max_elbow_drift: float = 0.0
    saw_top: bool = False
    saw_partial: bool = False
    perception_warnings: Counter = field(default_factory=Counter)
    rep_target: int = 0
    voice_flash: str = ""
    voice_flash_until: float = 0.0

    def start(self):
        self.active = True
        self.started_at = time.time()
        self.clean_reps = 0
        self.total_attempts = 0
        self.reps.clear()
        self.state = "ready"
        self.cue = "Find full extension"
        self.coach_tip = "Extend fully, then curl to shoulder height"
        self.last_coached_attempt = 0
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
            self.coach_tip = "Move so shoulder, elbow, and wrist are visible"
            self.perception_warnings["arm_not_visible"] += 1
            return

        if arm.confidence < KEYPOINT_CONFIDENCE:
            self.cue = "Low confidence"
            self.coach_tip = "Turn sideways and keep the working arm in frame"
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
                self.coach_tip = "Curl up smoothly when ready"
            elif angle < CURLING_ANGLE:
                self._begin_rep(arm)
                self.state = "curling"
                self.cue = "Curl higher"
                self.coach_tip = "Bring your wrist closer to shoulder height"

        elif self.state == "curling":
            if angle <= TOP_ANGLE:
                self.saw_top = True
                self.state = "top"
                self.cue = "Control down"
                self.coach_tip = "Good height. Lower slowly to full extension"
            elif angle <= PARTIAL_TOP_ANGLE:
                self.saw_partial = True
                self.cue = "Almost there"
                self.coach_tip = "A little higher before lowering"
            else:
                self.cue = "Curl higher"
                self.coach_tip = "Keep elbow planted and curl through the full range"

        elif self.state == "top":
            if angle > TOP_ANGLE + 15:
                self.state = "lowering"
                self.cue = "Full extension"
                self.coach_tip = "Lower until your arm is nearly straight"

        elif self.state == "lowering":
            if angle >= EXTENDED_ANGLE:
                self._finish_rep(arm, finished_with_full_extension=True)
            elif angle < CURLING_ANGLE - 10:
                self._finish_rep(arm, finished_with_full_extension=False)
            elif angle >= CURLING_ANGLE:
                self.cue = "Extend lower"
                self.coach_tip = "Finish the rep by fully extending at the bottom"

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
            self.coach_tip = "Clean rep. Keep that same tempo"
        else:
            self.cue = cue_for_issues(issues)
            self.coach_tip = fallback_rep_coach(issues, duration, self.min_angle, self.max_elbow_drift)

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


def fallback_rep_coach(issues, duration_seconds, min_angle, max_elbow_drift):
    if not issues:
        return "Clean rep. Keep that same tempo"
    if "partial_curl" in issues:
        return f"Curl higher before lowering. Best angle was {min_angle:.0f} deg"
    if "incomplete_extension" in issues:
        return "Return to a nearly straight arm before the next curl"
    if "elbow_drift" in issues:
        return f"Keep your elbow fixed. It drifted {max_elbow_drift:.0f}px"
    if "too_fast" in issues:
        return f"Slow down. That rep took {duration_seconds:.1f}s"
    return "Reset your arm and try one controlled full rep"


def rep_payload(session):
    if not session.reps:
        return None
    rep = session.reps[-1]
    return {
        "exercise": "one_arm_bicep_curl",
        "clean_reps": session.clean_reps,
        "total_attempts": session.total_attempts,
        "latest_rep": {
            "number": rep.number,
            "clean": rep.clean,
            "duration_seconds": rep.duration_seconds,
            "min_angle": rep.min_angle,
            "max_elbow_drift": rep.max_elbow_drift,
            "issues": rep.issues,
        },
    }


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


def put_wrapped_text(frame, text, origin, max_width, color, scale=0.55, thickness=1, line_gap=22):
    x, y = origin
    words = text.split()
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        (line_width, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if line and line_width > max_width:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y += line_gap
            line = word
        else:
            line = candidate
    if line:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return y


def draw_panel(frame, session, arm, fps, debug, llm_enabled):
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

    if session.rep_target > 0:
        rep_str = f"{str(session.clean_reps).zfill(2)}/{session.rep_target}"
        done = session.clean_reps >= session.rep_target
        rep_color = (0, 255, 120) if done else (255, 255, 255)
        cv2.putText(frame, f"TARGET {session.rep_target}", (36, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
    else:
        rep_str = str(session.clean_reps).zfill(2)
        rep_color = (255, 255, 255)
    cv2.putText(frame, rep_str, (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 3.2, rep_color, 7)
    cv2.putText(frame, "CLEAN REPS", (36, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 2)
    cv2.putText(frame, f"ATTEMPTS {session.total_attempts}", (38, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 2)

    if session.voice_flash and time.time() < session.voice_flash_until:
        cv2.putText(frame, session.voice_flash, (30, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    cue_color = (0, 255, 120) if session.cue in ("Ready", "Rep counted", "Set complete") else (0, 220, 255)
    cv2.putText(frame, session.cue.upper(), (30, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.72, cue_color, 2)
    cv2.putText(frame, "AI COACH", (30, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 210, 255), 1)
    put_wrapped_text(frame, session.coach_tip, (30, 410), panel_w - 60, (245, 245, 245), scale=0.55, thickness=1)

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
        min_angle = "--" if session.min_angle == 180.0 else f"{session.min_angle:.1f}"
        debug_lines = [
            f"state: {session.state}",
            f"arm: {side}",
            f"elbow angle: {angle}",
            f"min angle: {min_angle}",
            f"confidence: {conf}",
            f"attempts: {session.total_attempts}",
            f"llm: {'on' if llm_enabled else 'fallback'}",
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
    if not os.getenv("GROQ_API_KEY") and not coach_model_path():
        return fallback_coach(summary, question)

    sys_msg = (
        "You are GymBuddy, a quiet robotic fitness coach. Use only the measured JSON data. "
        "Do not invent injuries, weights, or hidden details. Give concise coaching: what went well, "
        "main issue, and one next-set correction. /no_think"
    )
    user_msg = f"Session JSON:\n{json.dumps(summary, indent=2)}\n\n"
    user_msg += f"User question: {question}\nAnswer in one sentence." if question else "Coach summary:"

    if os.getenv("GROQ_API_KEY"):
        return groq_coach([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}])

    # llama.cpp fallback
    global QWEN_MODEL
    try:
        from llama_cpp import Llama
    except ImportError:
        return fallback_coach(summary, question)
    with QWEN_LOCK:
        if QWEN_MODEL is None:
            QWEN_MODEL = Llama(model_path=coach_model_path(), n_ctx=4096, verbose=False)
        result = QWEN_MODEL(sys_msg + "\n\n" + user_msg, max_tokens=140, temperature=0.4, stop=["\n\n"])
    return result["choices"][0]["text"].strip()


def qwen_live_coach(payload):
    rep = payload["latest_rep"]
    if not os.getenv("GROQ_API_KEY") and not coach_model_path():
        return fallback_rep_coach(rep["issues"], rep["duration_seconds"], rep["min_angle"], rep["max_elbow_drift"])

    sys_msg = (
        "You are GymBuddy, a live bicep curl coach. Use only the measured JSON. "
        "Give ONE correction under 12 words. No JSON/sensor mentions. /no_think"
    )
    user_msg = f"Rep data:\n{json.dumps(payload, indent=2)}\nNext-rep cue:"

    if os.getenv("GROQ_API_KEY"):
        text = groq_coach(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_tokens=32, temperature=0.2,
        )
        return text.strip('"') or "Reset and try one controlled full rep"

    # llama.cpp fallback
    global QWEN_MODEL
    try:
        from llama_cpp import Llama
    except ImportError:
        return fallback_rep_coach(rep["issues"], rep["duration_seconds"], rep["min_angle"], rep["max_elbow_drift"])
    with QWEN_LOCK:
        if QWEN_MODEL is None:
            QWEN_MODEL = Llama(model_path=coach_model_path(), n_ctx=2048, verbose=False)
        result = QWEN_MODEL(sys_msg + "\n\n" + user_msg, max_tokens=32, temperature=0.2, stop=["\n", ". "])
    return result["choices"][0]["text"].strip().strip('"') or "Reset and try one controlled full rep"


def submit_live_coach(session, future):
    if session.total_attempts == 0 or session.last_coached_attempt == session.total_attempts:
        return future
    if future is not None and not future.done():
        return future

    payload = rep_payload(session)
    if payload is None:
        return future

    session.last_coached_attempt = session.total_attempts
    if coach_model_path():
        session.coach_tip = "AI coach is reading that rep..."
        return COACH_EXECUTOR.submit(qwen_live_coach, payload)

    rep = payload["latest_rep"]
    session.coach_tip = fallback_rep_coach(
        rep["issues"],
        rep["duration_seconds"],
        rep["min_angle"],
        rep["max_elbow_drift"],
    )
    return None


def collect_live_coach(session, future):
    if future is None or not future.done():
        return future
    try:
        session.coach_tip = future.result()
    except Exception as exc:
        session.coach_tip = f"LLM unavailable: {exc.__class__.__name__}. Using form cues."
    return None


def groq_coach(messages, max_tokens=140, temperature=0.4):
    """Blocking Groq call for set-review coaching."""
    resp = groq_client().chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


def session_context(session):
    parts = [
        f"clean_reps: {session.clean_reps}",
        f"total_attempts: {session.total_attempts}",
        f"rep_target: {session.rep_target if session.rep_target else 'not set'}",
        f"state: {session.state}",
        f"current cue: {session.cue}",
        f"current tip: {session.coach_tip}",
    ]
    return "\n".join(parts)


def handle_voice_command(session, cmd, value=0):
    """Mutate session from a parsed voice command. Returns spoken confirmation."""
    flash_duration = 3.0
    if cmd == "add_reps":
        session.rep_target = max(0, session.rep_target) + value
        msg = f"Done! Target is now {session.rep_target} reps."
        session.voice_flash = f"TARGET +{value} -> {session.rep_target}"
    elif cmd == "reduce_reps":
        old_target = max(0, session.rep_target)
        session.rep_target = max(0, old_target - value)
        msg = f"Done! Target reduced to {session.rep_target} reps."
        session.voice_flash = f"TARGET -{value} -> {session.rep_target}"
    elif cmd == "set_target":
        session.rep_target = value
        msg = f"Target set to {value} reps. Let's go!"
        session.voice_flash = f"TARGET: {value} REPS"
    elif cmd == "clear_target":
        session.rep_target = 0
        msg = "Target cleared."
        session.voice_flash = "TARGET CLEARED"
    elif cmd == "start":
        if not session.active:
            session.start()
        msg = "Set started. Give it your all!"
        session.voice_flash = "SET STARTED"
    elif cmd == "stop":
        if session.active:
            session.stop()
        msg = f"Nice work! {session.clean_reps} clean reps."
        session.voice_flash = f"SET DONE  {session.clean_reps} REPS"
    elif cmd == "reset":
        session.__init__()
        msg = "Session reset. Ready when you are."
        session.voice_flash = "SESSION RESET"
    else:
        return None
    session.voice_flash_until = time.time() + flash_duration
    return msg


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
    indexes = camera_indexes()
    camera = None
    active_index = indexes[0]
    for candidate_index in indexes:
        camera = open_camera_device(candidate_index)
        if camera is not None:
            active_index = candidate_index
            break

    if camera is None:
        print(camera_help(indexes))
        return

    session = CurlSession()
    debug = True
    last_time = time.time()
    fps = 0.0
    coach_future: Optional[Future] = None
    llm_enabled = bool(os.getenv("GROQ_API_KEY") or coach_model_path())

    print("GymBuddy Curl Coach started.")
    print(APP_VERSION)
    print("Press Space to start/stop a set. Press D for debug, R to reset, Q/Esc to quit.")
    if os.getenv("GROQ_API_KEY"):
        print("Groq cloud coaching enabled.")
    elif coach_model_path():
        print("Live LLM coaching enabled through GYMBUDDY_GGUF.")
    else:
        print("Live coaching is using fallback rules. Set GROQ_API_KEY or GYMBUDDY_GGUF.")

    voice = None
    if os.getenv("GROQ_API_KEY") and os.getenv("GYMBUDDY_VOICE", "1") != "0":
        try:
            from voice_assistant import VoiceAssistant
            voice = VoiceAssistant(
                get_context=lambda: session_context(session),
                on_command=lambda cmd, value=0: handle_voice_command(session, cmd, value),
            )
            voice.start()
        except Exception as exc:
            print(f"Voice assistant failed to start: {exc}")
            voice = None

    try:
        read_failures = 0
        last_camera_warning = 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                read_failures += 1
                now = time.time()
                if now - last_camera_warning >= CAMERA_RETRY_SECONDS:
                    print(camera_help([active_index]))
                    last_camera_warning = now
                if read_failures >= CAMERA_MAX_READ_FAILURES:
                    camera.release()
                    time.sleep(CAMERA_RETRY_SECONDS)
                    camera = open_camera_device(active_index)
                    read_failures = 0
                    if camera is None:
                        print(camera_help([active_index]))
                        time.sleep(CAMERA_RETRY_SECONDS)
                continue
            read_failures = 0

            result = model.predict(frame, conf=POSE_CONFIDENCE, verbose=False)[0]
            arm = select_arm(result)
            session.update(arm)
            coach_future = collect_live_coach(session, coach_future)
            coach_future = submit_live_coach(session, coach_future)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 0.001))
            last_time = now

            annotated_frame = frame.copy()
            draw_arm(annotated_frame, arm)
            draw_panel(annotated_frame, session, arm, fps, debug, llm_enabled)

            cv2.imshow("GymBuddy Curl Coach", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("r"):
                session = CurlSession()
                coach_future = None
            if key == 32:
                if session.active:
                    session.stop()
                    print_set_review(session)
                else:
                    session.start()
    finally:
        if voice is not None:
            voice.stop()
        camera.release()
        cv2.destroyAllWindows()
        COACH_EXECUTOR.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
