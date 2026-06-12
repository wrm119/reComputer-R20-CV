# SCRFD Face Detection on Raspberry Pi 5 + Hailo-8

This module packages SCRFD face-detection HEFs from the Hailo Model Zoo as a
one-command Docker demo for Raspberry Pi 5 / reComputer R20 with Hailo-8.

The module includes three 640x640x3 HEFs:

| Model | Path | Notes |
|---|---|---|
| SCRFD 500M | `model/scrfd_500m.hef` | Default, lightweight model |
| SCRFD 2.5G | `model/scrfd_2.5g.hef` | Balanced speed / accuracy |
| SCRFD 10G | `model/scrfd_10g.hef` | Larger and usually more accurate |

## Features

- Web preview for live face detection.
- Draws face boxes, confidence, and 5 facial landmarks.
- REST API returns boxes, scores, and `landmarks`.
- Supports demo video, USB camera, uploaded images/video frames, and offline video analysis.

## Files

| Path | Purpose |
|---|---|
| `web_detection.py` | FastAPI server, MJPEG preview, SCRFD post-process, offline video analysis |
| `model/*.hef` | SCRFD Hailo-8 models |
| `hailort-packages/*.whl` | HailoRT Python wheel matching the host driver major.minor version |
| `../../docker/hailo8/scrfd.dockerfile` | Dockerfile for this image |
| `video/test.mp4` | Built-in demo video |

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
cd src/rpi5_hailo8_scrfd

sudo docker build -f ../../docker/hailo8/scrfd.dockerfile \
    -t r20-hailo8-scrfd:latest .
```

## Run With Demo Video

Defaults to `model/scrfd_500m.hef`:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest
```

Open:

```text
http://<Pi5_IP>:8000
```

## Switch Models

Run SCRFD 2.5G:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest \
    python web_detection.py --model_path model/scrfd_2.5g.hef --video_path video/test.mp4
```

Run SCRFD 10G:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest \
    python web_detection.py --model_path model/scrfd_10g.hef --video_path video/test.mp4
```

## Run With USB Camera

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest \
    python web_detection.py --model_path model/scrfd_500m.hef --camera_id 0
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/scrfd/predict` | POST | Face detection on an uploaded image, a selected video frame, or the current camera frame |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or update confidence / SCRFD NMS thresholds |
| `/api/video/upload` | POST | Upload a video for offline analysis |
| `/api/video/analyze` | POST | Start offline analysis |
| `/api/video/status` | GET | Poll analysis progress |
| `/api/video/list` | GET | List uploaded and processed videos |
| `/api/video/download/{filename}` | GET | Download processed output |

Example:

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/scrfd/predict -F "file=@test.jpg"
```

## Post-Process

SCRFD HEFs do not use the YOLO on-chip NMS output layout. This module implements
SCRFD raw score / bbox / landmark post-processing in `post_process_hailo()`:

- stride 8 / 16 / 32 branch detection;
- distance-to-anchor-center bbox decode;
- 5-point landmark decode;
- confidence filtering and NMS;
- letterbox coordinate restoration to the original frame.
