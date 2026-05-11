# GymBuddy

This repository contains a simple Python camera object detector.

It uses the smallest current Ultralytics YOLO nano model, `yolo26n.pt`, for live object detection.

## Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Open the camera with object detection:

```bash
python3 open_camera.py
```

The first run downloads the model weights automatically. Press `q` or `Esc` to close the camera window.
