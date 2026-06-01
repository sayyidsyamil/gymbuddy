#!/bin/bash
OUTPUT=~/Desktop/gymbuddy_demo_$(date +%Y%m%d_%H%M%S).mp4
echo "Recording to: $OUTPUT"
echo "Press Q to stop."
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0 \
       -f pulse -i default \
       -c:v libx264 -preset ultrafast -crf 18 \
       -c:a aac -b:a 128k \
       "$OUTPUT"
echo "Saved to $OUTPUT"
