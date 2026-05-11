# gymbuddy_ros

ROS 1 (Noetic) implementation of the GymBuddy node graph from the project root README.

## Layout

- `msg/` — `Landmark`, `Skeleton`, `FormStatus`, `WorkoutStats`, `IntentUpdate`
- `scripts/` — 11 node entry points (one per node in the architecture table)
- `launch/gymbuddy.launch` — brings the whole graph up
- `requirements-ros.txt` — Python deps the nodes import (install into the same env rospy uses)

## Topic graph

| Topic                | Publisher                     | Subscriber                                   | Type                              |
|----------------------|-------------------------------|----------------------------------------------|-----------------------------------|
| `/raw_camera_frame`  | camera_input                  | pose_detection                               | `sensor_msgs/CompressedImage`     |
| `/skeleton_data`     | pose_detection                | form_analysis                                | `gymbuddy_ros/Skeleton`           |
| `/target_object_bbox`| pose_detection                | text_to_speech (placeholder)                 | `std_msgs/Int32MultiArray`        |
| `/form_status`       | form_analysis                 | rep_counter, text_to_speech                  | `gymbuddy_ros/FormStatus`         |
| `/workout_stats`     | rep_counter, workout_manager  | motivation, workout_manager, llm_decision    | `gymbuddy_ros/WorkoutStats`       |
| `/system_wake_state` | wake_word, speech_to_text     | speech_to_text                               | `std_msgs/Bool`                   |
| `/user_speech_raw`   | speech_to_text                | intent_extractor, llm_decision               | `std_msgs/String`                 |
| `/intent_update`     | intent_extractor              | workout_manager                              | `gymbuddy_ros/IntentUpdate`       |
| `/coaching_output`   | llm_decision, motivation, workout_manager | text_to_speech                   | `std_msgs/String`                 |

`workout_manager` re-publishes `/workout_stats` with `target_reps` populated; `motivation` and
`llm_decision` ignore the raw rep_counter stream (where `target_reps == 0`) and react to the
manager-annotated stream instead. This keeps the topic name in the architecture table while
avoiding a feedback loop.

## Install

In a Noetic catkin workspace:

```bash
cd ~/catkin_ws/src
ln -s /path/to/gymbuddy/gymbuddy_ros .
cd ~/catkin_ws && catkin_make
source devel/setup.bash

python3 -m pip install -r src/gymbuddy_ros/requirements-ros.txt
# optional LLM coach
export GYMBUDDY_QWEN_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf
```

## Run

```bash
roslaunch gymbuddy_ros gymbuddy.launch
```

Useful args:

```bash
roslaunch gymbuddy_ros gymbuddy.launch camera_device:=1 initial_target:=12
```

## Voice commands

Wake word is `hey jarvis` (closest model openwakeword ships out of the box — swap
the `wake_model` param for a custom "hey buddy" model when one is available).
After waking, say e.g.:

- "start set"
- "stop set"
- "set target to twelve reps"
- "add five reps"
- "reset"

Free-form questions ("how am I doing?") are routed to `llm_decision_node`, which
uses the Qwen GGUF if `GYMBUDDY_QWEN_GGUF` is set and falls back to a rule-based
reply otherwise.

## Notes

- `wake_word_node` and `speech_to_text_node` both open the default input device,
  but coordinate via `/system_wake_state`: wake_word goes silent for `cooldown_seconds`
  after a trigger, then STT records, transcribes, and publishes `False` to release.
- `pose_detection_node` uses MediaPipe (33 landmarks) per the architecture spec, not
  the YOLO pose model in `open_camera.py`. The standalone YOLO script remains usable.
