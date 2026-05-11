# How to Run the GymBuddy ROS Nodes

Step-by-step guide to bring up the full 11-node graph from
[gymbuddy_ros/](gymbuddy_ros/).

---

## 1. Host requirements

ROS 1 Noetic only runs cleanly on **Ubuntu 20.04** (or 20.04 inside WSL2 on
Windows). The rest of this guide assumes you're on that.

| Need              | How to check / install                                                |
| ----------------- | --------------------------------------------------------------------- |
| Ubuntu 20.04      | `lsb_release -a`                                                      |
| ROS Noetic        | `printenv ROS_DISTRO` should print `noetic`. Install: [ros.org/install](http://wiki.ros.org/noetic/Installation/Ubuntu) |
| Python 3.8+       | `python3 --version`                                                   |
| Webcam            | `ls /dev/video*` should list at least `/dev/video0`                   |
| Microphone        | `arecord -l` should list at least one capture device                  |
| Speaker / output  | `aplay -l` should list at least one playback device                   |

If you're on **WSL2**, USB webcams and audio require extra setup — usbipd-win
for the camera and PulseAudio forwarding for mic/speakers. Native Ubuntu is
much easier for a first run.

---

## 2. Create a catkin workspace and link the package

If you don't already have one:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Symlink this repo's package into your workspace:

```bash
ln -s ~/path/to/gymbuddy/gymbuddy_ros ~/catkin_ws/src/gymbuddy_ros
```

(Use the absolute path to wherever you cloned `gymbuddy`.)

---

## 3. Build the package (generates the custom messages)

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
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

## 4. Install Python dependencies

System packages first (mediapipe and sounddevice need them):

```bash
sudo apt install -y portaudio19-dev espeak ffmpeg
```

Then the Python deps:

```bash
python3 -m pip install -r ~/catkin_ws/src/gymbuddy_ros/requirements-ros.txt
```

This installs: `mediapipe`, `sounddevice`, `openwakeword`, `faster-whisper`,
`pyttsx3`, `llama-cpp-python`, plus `opencv-python` and `numpy`.

The first time `openwakeword` runs it will download its small wake-word model
(`hey_jarvis`) automatically. The first time `faster-whisper` runs it will
download the `tiny.en` model (~75 MB).

---

## 5. (Optional) Local LLM coach

If you want the Qwen-backed `llm_decision_node` instead of the rule-based
fallback, point it at a GGUF file:

```bash
export GYMBUDDY_QWEN_GGUF=/absolute/path/to/Qwen3.5-0.8B-Q4_K_M.gguf
```

Put this in your `~/.bashrc` if you want it persistent. Without this env var
the node still works — it just gives canned summaries.

---

## 6. Launch the whole graph

In one terminal, start the ROS master and all 11 nodes:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch gymbuddy_ros gymbuddy.launch
```

Useful arguments:

```bash
# pick a different camera and a 12-rep target
roslaunch gymbuddy_ros gymbuddy.launch camera_device:=1 initial_target:=12
```

You should see each node log a "ready" line. The first launch will be slow
(MediaPipe, openwakeword, faster-whisper all download/initialize models).

---

## 7. Verify the nodes are alive

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

# elbow angle / form status
rostopic echo /form_status

# rep counts
rostopic echo /workout_stats

# anything the coach says
rostopic echo /coaching_output
```

---

## 8. Use it

### Voice control

Say the wake word ("hey jarvis" — see note in
[gymbuddy_ros/README.md](gymbuddy_ros/README.md)) then a command:

- "start set"
- "stop set"
- "set target to twelve reps"
- "add five reps"
- "reset"
- Any free-form question goes to the LLM coach (e.g. "how am I doing")

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

This is the fastest way to test the vision pipeline without setting up a mic.

---

## 9. Run nodes individually (for debugging)

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

## 10. Common problems

**"ImportError: No module named gymbuddy_ros.msg"**
You didn't `source devel/setup.bash` in the terminal you're using.

**`rostopic list` doesn't show `/raw_camera_frame`**
The camera node failed silently. Run it standalone (`rosrun gymbuddy_ros
camera_input_node.py`) and read its output.

**MediaPipe install fails on Python 3.10+**
MediaPipe wheels are picky. Use Python 3.8 or 3.9 (Ubuntu 20.04 default).

**`sounddevice.PortAudioError: Error querying device -1`**
No default audio device. List devices with
`python3 -c "import sounddevice; print(sounddevice.query_devices())"` and
either set a default in PulseAudio or pass `device=` into the InputStream
calls in [wake_word_node.py](gymbuddy_ros/scripts/wake_word_node.py) and
[speech_to_text_node.py](gymbuddy_ros/scripts/speech_to_text_node.py).

**Wake word never fires**
Lower the threshold:
`rosrun gymbuddy_ros wake_word_node.py _threshold:=0.3`

**`pyttsx3` says nothing**
On Linux it needs `espeak` (installed in step 4). Test with
`espeak "hello"`.

**Two nodes fight for the microphone**
That's expected design — `wake_word_node` releases the mic for
`cooldown_seconds` (default 8s) after a trigger so `speech_to_text_node` can
record. If the timing feels off, tune `cooldown_seconds` and the STT node's
`max_seconds`/`silence_seconds` params.

---

## 11. Shutdown

`Ctrl-C` in the `roslaunch` terminal stops every node cleanly.
