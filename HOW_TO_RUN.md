# How to Run the GymBuddy ROS Nodes

Step-by-step guide to bring up the full 10-node graph from
[gymbuddy_ros/](gymbuddy_ros/).

---

## 1. Host requirements

ROS 1 Noetic runs most cleanly on **Ubuntu 20.04** (or 20.04 inside WSL2 on
Windows). The voice pipeline records and plays audio through PulseAudio
(`arecord`/`aplay`), so it is Linux-specific; the vision pipeline also works on
macOS via RoboStack for development.

| Need              | How to check / install                                                |
| ----------------- | --------------------------------------------------------------------- |
| Ubuntu 20.04      | `lsb_release -a`                                                      |
| ROS Noetic        | `printenv ROS_DISTRO` should print `noetic`. Install: [ros.org/install](http://wiki.ros.org/noetic/Installation/Ubuntu) |
| Python 3.8+       | `python3 --version`                                                   |
| Webcam            | `ls /dev/video*` should list at least `/dev/video0`                   |
| Microphone        | `arecord -l` should list at least one capture device                  |
| Speaker / output  | `aplay -l` should list at least one playback device                   |
| Groq API key      | `echo $GROQ_API_KEY` — required for speech-to-text, text-to-speech, and intent parsing |

If you're on **WSL2**, USB webcams and audio require extra setup — usbipd-win
for the camera and PulseAudio forwarding for mic/speakers. Native Ubuntu is
much easier for a first run.

### macOS / RoboStack

If you installed Noetic with RoboStack:

```bash
mamba activate ros_env
```

Use `source ~/catkin_ws/devel/setup.zsh` instead of the Ubuntu
`source /opt/ros/noetic/setup.bash` line. Camera access is controlled by macOS
Privacy settings. Note that `voice_io_node` uses `arecord`/`aplay`, which are
Linux/PulseAudio tools — run with `use_voice:=false use_tts:=false` on macOS, or
drive commands via `rostopic pub` (see step 7).

---

## 2. Create a catkin workspace and link the package

If you don't already have one:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash   # Ubuntu/Linux bash
# or: source devel/setup.zsh   # macOS/zsh
```

Symlink this repo's package into your workspace:

```bash
ln -s /path/to/gymbuddy/gymbuddy_ros ~/catkin_ws/src/gymbuddy_ros
```

Ensure all node scripts are executable (only needs to be done once):

```bash
chmod +x ~/catkin_ws/src/gymbuddy_ros/scripts/*.py
```

---

## 3. Build the package (generates the custom messages)

```bash
cd ~/catkin_ws
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash   # Ubuntu/Linux bash
# or: source devel/setup.zsh   # macOS/zsh
```

You should see lines like:

```
[ 50%] Generating Python from MSG gymbuddy_ros/Skeleton
[ 75%] Generating Python from MSG gymbuddy_ros/WorkoutStats
...
```

Verify the messages are importable:

```bash
python3 -c "from gymbuddy_ros.msg import Skeleton, FormStatus, WorkoutStats; print('ok')"
```

If this fails, you forgot to `source devel/setup.bash` in your current shell.

---

## 4. Install dependencies

System packages first (MediaPipe and the PulseAudio audio tools):

```bash
sudo apt install -y alsa-utils pulseaudio ffmpeg
```

Then the Python deps (from this repo's root `requirements.txt`):

```bash
python3 -m pip install -r /path/to/gymbuddy/requirements.txt
```

The ROS nodes need the "Core" block: `opencv-python`, `numpy`, `mediapipe`,
`groq`, and `pynput`. The "Prototypes" block (`ultralytics`, `llama-cpp-python`)
is only used by the standalone scripts in `tests/` and can be skipped.
The first time MediaPipe runs it downloads its small pose model automatically.

---

## 5. Set your Groq API key

`voice_io_node` (STT + TTS) and `intent_extractor_node` (intent LLM) call the
Groq cloud API. Export your key before launching:

```bash
export GROQ_API_KEY=your_key_here
```

The repo's [`start.sh`](start.sh) does this for you by sourcing a gitignored
`secrets.sh` — copy the template and fill it in:

```bash
cp secrets.sh.example secrets.sh   # then edit secrets.sh
```

Without the key, `voice_io_node` will not start and `intent_extractor_node`
falls back to regex-only command matching.

---

## 6. Launch the whole graph

Easiest path (sources ROS, loads `secrets.sh`, then launches):

```bash
./start.sh
```

Or manually, in one terminal:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export GROQ_API_KEY=your_key_here
roslaunch gymbuddy_ros gymbuddy.launch
```

Useful arguments:

```bash
# pick a different camera and a 12-rep target
roslaunch gymbuddy_ros gymbuddy.launch camera_device:=1 initial_target:=12

# test the full form/count pipeline without a camera
roslaunch gymbuddy_ros gymbuddy.launch use_camera:=false use_sim_skeleton:=true initial_target:=3

# avoid microphone/speaker nodes while debugging vision
roslaunch gymbuddy_ros gymbuddy.launch use_voice:=false use_tts:=false
```

You should see each node log a "ready" line. The first launch is a little slow
while MediaPipe initialises.

---

## 7. Use it

### On-screen display

The `display_node` window opens on a **home screen**: click an exercise card,
set the target reps with `-`/`+`, and click **START**. In the live workout view:

- `1` / `2` — switch between Bicep Curl and Lateral Raise
- `D` — toggle the debug overlay
- `Q` — return to the home screen
- `Esc` — quit

### Voice control (push-to-talk)

Hold **`SPACE`**, speak a command, then release. The clip is transcribed by Groq
Whisper and classified by `intent_extractor_node`. Recognised commands:

- "start set" / "stop set"
- "reset"
- "set target to twelve reps"
- "add five reps" / "remove three reps"

### Manual control (without voice)

You can poke the same intents directly via `rostopic pub`:

```bash
# start a set
rostopic pub -1 /intent_update gymbuddy_ros/IntentUpdate \
  "{header: {stamp: now}, action: 'start_set', value: 0, text: ''}"

# change target to 12
rostopic pub -1 /intent_update gymbuddy_ros/IntentUpdate \
  "{header: {stamp: now}, action: 'update_target', value: 12, text: ''}"

# stop a set
rostopic pub -1 /intent_update gymbuddy_ros/IntentUpdate \
  "{header: {stamp: now}, action: 'stop_set', value: 0, text: ''}"
```

This is the fastest way to test the vision pipeline without a mic or API key.

---

## 8. Monitor the voice pipeline (live)

Open a new terminal and tail the `voice_io` and `intent_extractor` logs:

```bash
tail -f ~/.ros/log/latest/voice_io-*.log \
         ~/.ros/log/latest/intent_extractor-*.log \
  | grep --line-buffered -E "SPACE|STT:|intent|Groq|warn|error"
```

What to expect:
- `voice_io: SPACE held — recording...` — push-to-talk started
- `STT: add five reps` — Whisper transcription
- `intent_extractor: add_reps value=5` — the command the LLM/regex classified

Alternatively, listen directly to the ROS topics (one per terminal):

```bash
# raw Whisper transcription
rostopic echo /user_speech_raw

# classified intent sent to the workout manager
rostopic echo /intent_update

# anything spoken by the coach
rostopic echo /coaching_output
```

---

## 9. Verify the nodes are alive

In a second terminal (always re-source first):

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

List the running nodes and topics:

```bash
rosnode list
rostopic list
```

Watch a few key topics:

```bash
# camera frames are flowing
rostopic hz /raw_camera_frame              # should be ~30 Hz

# pose detection is working
rostopic hz /skeleton_data                 # ~15-30 Hz when you're in frame

# elbow / shoulder angle and form status
rostopic echo /form_status

# rep counts
rostopic echo /workout_stats
```

---

## 10. Run nodes individually (for debugging)

You don't have to launch everything. To run just one node:

```bash
rosrun gymbuddy_ros camera_input_node.py
rosrun gymbuddy_ros pose_detection_node.py
rosrun gymbuddy_ros form_analysis_node.py
rosrun gymbuddy_ros rep_counter_node.py
# etc.
```

You always need `roscore` running first if you're not using `roslaunch`:

```bash
roscore
```

---

## 11. Common problems

**"ImportError: No module named gymbuddy_ros.msg"**
You didn't `source devel/setup.bash` in the terminal you're using.

**`voice_io: GROQ_API_KEY not set — node will not start`**
Export `GROQ_API_KEY` (step 5) before launching, or use `./start.sh`.

**`rostopic list` doesn't show `/raw_camera_frame`**
The camera node failed silently. Run it standalone (`rosrun gymbuddy_ros
camera_input_node.py`) and read its output. On macOS, enable Camera permission
for the terminal app and rerun.

**Need to test without camera access**
Use the simulator:
`roslaunch gymbuddy_ros gymbuddy.launch use_camera:=false use_sim_skeleton:=true`.

**MediaPipe install fails on Python 3.10+**
MediaPipe wheels are picky. Use Python 3.8 or 3.9 (Ubuntu 20.04 default).

**Push-to-talk does nothing / no transcription**
`pynput` needs an active keyboard focus and (on some setups) an X session;
check the `voice_io` log for `SPACE held — recording...`. Confirm `arecord -D
pulse -d 2 test.wav` records and `aplay -D pulse test.wav` plays back.

**No audio plays back**
TTS uses `aplay -D pulse`. Verify PulseAudio is running and a default sink is
set (`pactl info`).

**Intents are not recognised / "regex only" in the log**
`intent_extractor_node` falls back to regex without `GROQ_API_KEY`. Regex still
handles the basic commands; set the key for free-form phrasing.

---

## 12. Shutdown

`Ctrl-C` in the `roslaunch` (or `./start.sh`) terminal stops every node cleanly.
