# GymBuddy

GymBuddy is a ROS 1 (Noetic) workout coach for dumbbell **bicep curls** and
**lateral raises**. A camera drives live pose estimation, a dual-arm state
machine counts clean reps and flags form issues, and a push-to-talk voice
interface lets you control the session hands-free.

The full node graph and topic schema live in
[gymbuddy_ros/](gymbuddy_ros/). A step-by-step bring-up guide is in
[HOW_TO_RUN.md](HOW_TO_RUN.md).

## Features

- Two exercises: **bicep curl** (elbow flexion) and **lateral raise** (shoulder abduction)
- **Dual-arm rep counting** — both arms must complete the full range of motion for a rep to count
- Live form cues: partial curl/raise, incomplete extension, no return, fast reps, low confidence
- On-screen **display** with a home screen (pick exercise + target reps) and a live workout view
- **Push-to-talk voice control** — hold `SPACE`, speak a command, release
- Spoken coaching and milestone encouragement via text-to-speech
- Per-set calorie estimate (MET-based) and a post-set summary

## Architecture

GymBuddy is a graph of single-purpose ROS nodes in
[`gymbuddy_ros/scripts/`](gymbuddy_ros/scripts/):

| Node | Role |
|------|------|
| `camera_input_node`    | Captures frames from a USB webcam or an Astra colour topic; publishes JPEG frames |
| `pose_detection_node`  | MediaPipe Pose (33 landmarks → COCO-17) → `/skeleton_data` |
| `form_analysis_node`   | Computes joint angles per arm and publishes `/form_status` |
| `rep_counter_node`     | Dual-arm state machine; counts clean reps and tags form issues |
| `workout_manager_node` | Central state machine: active exercise, target reps, set history, calories |
| `motivation_node`      | Logic-driven encouragement every Nth rep |
| `intent_extractor_node`| Turns transcribed speech into workout commands (Groq LLM, regex fallback) |
| `voice_io_node`        | Push-to-talk speech-to-text + text-to-speech (Groq Whisper + Orpheus) |
| `display_node`         | OpenCV GUI — home screen + live workout view |
| `sim_skeleton_node`    | Optional synthetic skeleton publisher for testing without a camera |

See [gymbuddy_ros/README.md](gymbuddy_ros/README.md) for the full topic graph
(publishers, subscribers, and message types).

## Requirements

- Ubuntu 20.04 + ROS 1 Noetic (a catkin workspace)
- Python 3.8+
- A webcam, microphone, and speaker
- A **Groq API key** for speech-to-text, text-to-speech, and intent parsing

Set the API key before launching. The repo ships a template — copy it and fill
in your key (the real `secrets.sh` is gitignored):

```bash
cp secrets.sh.example secrets.sh   # then edit secrets.sh with your GROQ_API_KEY
```

Without `GROQ_API_KEY`, `intent_extractor_node` falls back to regex command
matching and the Groq-based voice I/O is unavailable.

## Install

Build the package in a Noetic catkin workspace and install the Python deps —
the detailed walkthrough (system packages, message generation, troubleshooting)
is in [HOW_TO_RUN.md](HOW_TO_RUN.md):

```bash
cd ~/catkin_ws/src
ln -s /path/to/gymbuddy/gymbuddy_ros .
cd ~/catkin_ws && catkin_make && source devel/setup.bash
python3 -m pip install -r /path/to/gymbuddy/requirements.txt
```

## Run

The quickest path is the helper script, which sources ROS, loads `secrets.sh`,
and launches the graph:

```bash
./start.sh
```

Or launch directly:

```bash
roslaunch gymbuddy_ros gymbuddy.launch
```

Useful launch arguments:

```bash
# pick a camera and a 12-rep target
roslaunch gymbuddy_ros gymbuddy.launch camera_device:=1 initial_target:=12

# run the full pipeline without a camera (synthetic skeleton)
roslaunch gymbuddy_ros gymbuddy.launch use_camera:=false use_sim_skeleton:=true

# disable voice/TTS while debugging vision
roslaunch gymbuddy_ros gymbuddy.launch use_voice:=false use_tts:=false
```

## Using it

**On the display:** the home screen lets you click an exercise card, set the
target reps with `-`/`+`, and press **START**. In the workout view, keys `1`/`2`
switch exercise, `D` toggles the debug overlay, `Q` returns home, and `Esc`
quits.

**By voice:** hold `SPACE`, speak, then release. Recognised commands:

- "start set" / "stop set"
- "reset"
- "set target to twelve reps"
- "add five reps" / "remove three reps"

You can also drive the same commands without a mic by publishing
`gymbuddy_ros/IntentUpdate` on `/intent_update` — see the "Use it" section in
[HOW_TO_RUN.md](HOW_TO_RUN.md).

## Camera setup

Use a side view with your shoulder, elbow, and wrist visible. Rep counting uses
body pose, not dumbbell detection, so a dumbbell is optional.

## Prototypes

The pre-ROS experiments that informed this build live in
[`tests/`](tests/) — `open_camera.py` (standalone YOLO pose + curl counter),
`test_curl_logic.py` (its unit tests), and `voice_assistant.py` / `voice_buddy.py`
(early voice prototypes). They are not part of the ROS graph.
