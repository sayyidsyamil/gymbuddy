# gymbuddy_ros

ROS 1 (Noetic) implementation of the GymBuddy node graph. See the
[project root README](../README.md) for the high-level overview and
[HOW_TO_RUN.md](../HOW_TO_RUN.md) for full bring-up steps.

## Layout

- `msg/` — `Landmark`, `Skeleton`, `FormStatus`, `WorkoutStats`, `IntentUpdate`
- `scripts/` — 10 node entry points
- `launch/gymbuddy.launch` — brings the whole graph up

Python deps live in the repo-root [`requirements.txt`](../requirements.txt)
(install into the same env rospy uses).

## Topic graph

| Topic               | Publisher                                  | Subscriber                              | Type                          |
|---------------------|--------------------------------------------|-----------------------------------------|-------------------------------|
| `/raw_camera_frame` | camera_input                               | pose_detection, display                 | `sensor_msgs/CompressedImage` |
| `/skeleton_data`    | pose_detection, sim_skeleton               | form_analysis, display                  | `gymbuddy_ros/Skeleton`       |
| `/active_exercise`  | display                                    | form_analysis, rep_counter              | `std_msgs/String`             |
| `/form_status`      | form_analysis                              | rep_counter, display                    | `gymbuddy_ros/FormStatus`     |
| `/workout_stats`    | rep_counter, workout_manager               | workout_manager, motivation, display    | `gymbuddy_ros/WorkoutStats`   |
| `/user_speech_raw`  | voice_io                                   | intent_extractor                        | `std_msgs/String`             |
| `/intent_update`    | intent_extractor, display, workout_manager, sim_skeleton | rep_counter, workout_manager, sim_skeleton | `gymbuddy_ros/IntentUpdate` |
| `/coaching_output`  | motivation, workout_manager                | voice_io, display                       | `std_msgs/String`             |
| `/tts_priority`     | motivation                                 | voice_io, display                       | `std_msgs/String`             |

`workout_manager` subscribes to `/workout_stats`, then re-publishes it with `target_reps`
populated. To avoid a feedback loop, `workout_manager` only acts on the raw rep_counter
stream (`target_reps == 0`), while `motivation` only reacts to the manager-annotated stream
(`target_reps != 0`). `display` shows both.

## Install

In a Noetic catkin workspace:

```bash
cd ~/catkin_ws/src
ln -s /path/to/gymbuddy/gymbuddy_ros .
cd ~/catkin_ws && catkin_make
source devel/setup.bash

python3 -m pip install -r /path/to/gymbuddy/requirements.txt
```

Speech-to-text, text-to-speech, and intent parsing call the Groq cloud API, so export a key
first (the repo's `start.sh` sources `secrets.sh` for this — see the root README):

```bash
export GROQ_API_KEY=your_key_here
```

Without `GROQ_API_KEY`, `voice_io_node` refuses to start and `intent_extractor_node` falls
back to regex-only command matching.

## Run

```bash
roslaunch gymbuddy_ros gymbuddy.launch
```

Useful args:

```bash
roslaunch gymbuddy_ros gymbuddy.launch camera_device:=1 initial_target:=12
roslaunch gymbuddy_ros gymbuddy.launch use_camera:=false use_sim_skeleton:=true
roslaunch gymbuddy_ros gymbuddy.launch use_voice:=false use_tts:=false
```

## Voice commands

`voice_io_node` is **push-to-talk**: hold `SPACE`, speak, then release. The audio is sent to
Groq Whisper for transcription and published on `/user_speech_raw`. Recognised commands:

- "start set" / "stop set"
- "reset"
- "set target to twelve reps"
- "add five reps" / "remove three reps"

`intent_extractor_node` classifies the utterance with a Groq LLM (regex fallback) and emits a
`gymbuddy_ros/IntentUpdate` on `/intent_update`.

## Notes

- `voice_io_node` records via `arecord` on the PulseAudio device (avoids the PortAudio/ALSA
  issues the older sounddevice path hit) and speaks coaching text back through Groq TTS (Orpheus).
- `pose_detection_node` uses MediaPipe (33 landmarks, remapped to a COCO-17 layout), not the
  YOLO pose model in `tests/open_camera.py`. The standalone YOLO script remains usable.
