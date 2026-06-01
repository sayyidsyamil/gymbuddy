#!/bin/bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
# Load local secrets (GROQ_API_KEY). See secrets.sh.example
source "$(dirname "$0")/secrets.sh"
export DISPLAY=:0
roslaunch gymbuddy_ros gymbuddy.launch
