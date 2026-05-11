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


ROS 1 Architecture

**1\. Functional Modules & Nodes**

### **Posture Detection (Vision Pipeline)**

* **camera\_input\_node**: High-frequency frame capture (USB/CSI).  
* **pose\_detection\_node**: MediaPipe inference to generate 33-point body landmark coordinates.  
* **form\_analysis\_node**: Calculates geometric vectors (e.g., knee angles, spine alignment).  
* **rep\_counter\_node**: Logic gate that increments counts based on Range of Motion (ROM) and form criteria.

### **Speech & NLP (Interaction Pipeline)**

* **wake\_word\_node**: Low-power listener for "Hey Buddy."  
* **speech\_to\_text\_node**: Transcribes audio buffers into raw text strings.  
* **intent\_extractor\_node**: Parses commands (e.g., "Add 5 reps") into actionable JSON updates.

### **Intelligence & Logic**

* **llm\_decision\_node**: High-level reasoning. Estimates calories ($MET \\times kg \\times hr$) and generates context-aware coaching.  
* **workout\_manager\_node**: The central state machine tracking exercise types, set history, and user targets.  
* **motivation\_node**: Logic-driven encouragement triggered by performance milestones (e.g., every 3rd rep).

### **Feedback Output**

* **text\_to\_speech\_node**: Converts text to audio with an managed queue to prevent overlapping messages.

## ---

**2\. Topic Communication Schema**

| Topic | Publisher | Subscriber | Data Type / Example |
| :---- | :---- | :---- | :---- |
| /raw\_camera\_frame | camera\_input\_node | pose\_detection\_node | Compressed JPEG / Raw Image |
| /skeleton\_data | pose\_detection\_node | form\_analysis\_node | 33 Landmark XYZ Coordinates |
| /target\_object\_bbox | pose\_detection\_node | text\_to\_speech\_node | \[x\_min, y\_min, x\_max, y\_max\] |
| /form\_status | form\_analysis\_node | rep\_counter\_node, text\_to\_speech\_node | "back\_rounded", "depth\_reached" |
| /workout\_stats | rep\_counter\_node | motivation\_node, workout\_manager\_node | Int (e.g., 3, 6, 9\) |
| /system\_wake\_state | wake\_word\_node | speech\_to\_text\_node | Boolean (True/False) |
| /user\_speech\_raw | speech\_to\_text\_node | llm\_decision\_node, intent\_extractor\_node | "Add 5 more reps" |
| /intent\_update | intent\_extractor\_node | workout\_manager\_node | {"action": "update\_target", "value": 12} |
| /coaching\_output | llm\_decision\_node, motivation\_node | text\_to\_speech\_node | "Great job, keep going\!" |

## ---

**3\. System Flow (Agent Logic)**

The system operates through three primary concurrent loops:

### **A. The Biomechanical Loop (Vision-to-Count)**

1. **Capture**: camera\_input\_node streams frames to /raw\_camera\_frame.  
2. **Detection**: pose\_detection\_node extracts landmarks and publishes to /skeleton\_data.  
3. **Analysis**: form\_analysis\_node evaluates angles. If form is incorrect, it sends a violation to /form\_status (triggering immediate TTS feedback).  
4. **Counting**: If form is correct and ROM is satisfied, rep\_counter\_node increments the count and publishes to /workout\_stats.

### **B. The Command Loop (Voice-to-Action)**

1. **Trigger**: wake\_word\_node detects "Hey Buddy" and flips /system\_wake\_state to True.  
2. **Transcription**: speech\_to\_text\_node listens, transcribes, and publishes to /user\_speech\_raw.  
3. **Parsing**:  
   * intent\_extractor\_node pulls specific parameters (e.g., "Add 5") and updates the workout\_manager\_node.  
   * llm\_decision\_node interprets complex queries (e.g., "How many calories?") and sends text to /coaching\_output.

### **C. The Feedback Loop (Stats-to-Speech)**

1. **Evaluation**: workout\_manager\_node compares current /workout\_stats against the target goal.  
2. **Encouragement**: Every 3 reps, motivation\_node generates a "Keep going\!" message via /coaching\_output.  
3. **Execution**: text\_to\_speech\_node synthesizes all incoming text from /coaching\_output and /form\_status into a sequential audio queue.