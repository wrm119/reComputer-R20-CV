# YOLOv8 Pose on Raspberry Pi 5 + Hailo-8

This module packages YOLOv8 Pose as a one-command Docker deployment for
Raspberry Pi 5 / reComputer R20 with Hailo-8.

Bundled HEFs:

| Model | Path | Notes |
|---|---|---|
| YOLOv8s Pose | `model/yolov8s_pose.hef` | Default, speed-oriented |
| YOLOv8m Pose | `model/yolov8m_pose.hef` | Larger, usually more accurate and slower |

## Features

- Real-time web preview for human pose estimation.
- Draws person boxes, confidence, COCO 17 keypoints, and skeleton lines.
- REST API returns boxes, scores, and `keypoints`.
- Supports demo video, USB camera, uploaded images/video frames, and offline video analysis.

## Prerequisites

### Install Docker

Run the following commands on the development board to install Docker:

```bash
# Download installation script
curl -fsSL https://get.docker.com -o get-docker.sh
# Install using Aliyun mirror source
sudo sh get-docker.sh --mirror Aliyun
# Start Docker and enable auto-start on boot
sudo systemctl enable docker
sudo systemctl start docker
```

## Build

```bash
cd src/rpi5_hailo8_yolov8_pose

sudo docker build -f ../../docker/hailo8/yolov8_pose.dockerfile \
    -t r20-hailo8-yolov8_pose:latest .
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8_pose:latest
```

Open:

```text
http://<Pi5_IP>:8000
```

## USB Camera

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8_pose:latest \
    python web_detection.py --model_path model/yolov8s_pose.hef --camera_id 0
```

## Switch To YOLOv8m Pose

```bash
python web_detection.py --model_path model/yolov8m_pose.hef --video_path video/test.mp4
```

## API

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8_pose/predict -F "file=@test.jpg"
```

Each prediction includes:

```json
{
  "class": "person",
  "confidence": 0.91,
  "box": {"x1": 120, "y1": 80, "x2": 360, "y2": 520},
  "keypoints": [{"x": 180, "y": 120, "score": 0.87}]
}
```

## Post-processing

`post_process_hailo()` supports the Hailo Model Zoo raw YOLOv8-pose heads
(`bbox`, `score`, `keypoints` at 20/40/80 feature maps) and also keeps a fallback
for NMS-style rows such as:

```text
[ymin, xmin, ymax, xmax, score, kpt0_x, kpt0_y, kpt0_score, ..., kpt16_x, kpt16_y, kpt16_score]
```

The first inference prints every raw output tensor shape. Use that log if the
HailoRT pose layout needs a small adjustment on-device.
