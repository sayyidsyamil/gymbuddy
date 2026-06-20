cmake_minimum_required(VERSION 3.0.2)
project(gymbuddy_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  Landmark.msg
  Skeleton.msg
  FormStatus.msg
  WorkoutStats.msg
  IntentUpdate.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/camera_input_node.py
  scripts/pose_detection_node.py
  scripts/form_analysis_node.py
  scripts/rep_counter_node.py
  scripts/voice_io_node.py
  scripts/intent_extractor_node.py
  scripts/workout_manager_node.py
  scripts/motivation_node.py
  scripts/display_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
