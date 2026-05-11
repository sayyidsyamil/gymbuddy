# GymBuddy

GymBuddy is a quiet human-computer interaction prototype for one-arm bicep curls.

It uses live pose estimation for rep counting and form cues. The LLM coach is optional and only runs after a set, using structured workout metrics instead of raw video.

## Features

- One-arm side-view curl tracking
- Clean rep counting with an elbow-angle state machine
- Live cues for partial curls, incomplete extension, elbow drift, fast reps, low confidence, and arm visibility
- Debug overlay for state, selected arm, elbow angle, confidence, attempts, and FPS
- Post-set coaching summary
- Optional local Qwen3.5-0.8B GGUF coaching through `llama-cpp-python`

## Install

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Optional LLM support:

```bash
python3 -m pip install -r requirements-llm.txt
export GYMBUDDY_QWEN_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf
```

If `GYMBUDDY_QWEN_GGUF` is not set, GymBuddy uses a built-in rule-based coach summary.

## Run

```bash
python3 open_camera.py
```

The first run downloads `yolo26n-pose.pt` automatically.

## Controls

- `Space`: start or stop a set
- `D`: toggle debug overlay
- `R`: reset
- `Q` or `Esc`: quit

After stopping a set, the terminal prints a structured set review and lets you type follow-up questions for the coach.

## Camera Setup

Use a one-arm side view. Make sure your shoulder, elbow, and wrist are visible. A dumbbell is optional; rep counting uses body pose, not dumbbell detection.
