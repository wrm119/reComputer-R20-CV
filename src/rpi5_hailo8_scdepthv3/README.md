# SCDepthV3 Depth Estimation on Raspberry Pi 5 + Hailo-8

This module packages `scdepthv3.hef` as a one-command Docker deployment for
Raspberry Pi 5 / reComputer R20 with Hailo-8.

## Features

- Real-time web preview for monocular depth estimation.
- Decodes the model output into a relative-depth map.
- Displays the result as a pseudo-color depth overlay.
- REST API route: `/api/models/scdepthv3/predict`.
- API response keeps the YOLO-style `predictions` list and adds depth stats.

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
cd src/rpi5_hailo8_scdepthv3

sudo docker build -f ../../docker/hailo8/scdepthv3.dockerfile \
    -t r20-hailo8-scdepthv3:latest .
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scdepthv3:latest
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
    r20-hailo8-scdepthv3:latest \
    python web_detection.py --model_path model/scdepthv3.hef --camera_id 0
```

## API

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/scdepthv3/predict -F "file=@test.jpg"
```

Each response includes a YOLO-style `predictions` list:

```json
{
  "class": "depth",
  "confidence": 1.0,
  "min_depth": 0.12,
  "max_depth": 8.4,
  "mean_depth": 1.9,
  "median_depth": 1.5,
  "valid_pixels": 921600
}
```

Depth values are relative, not metric meters.

## Post-processing

The raw model output is converted to relative depth with:

```text
depth = 1 / (sigmoid(raw) * 10 + 0.009)
```

The first inference prints the raw output shape and decoded depth statistics.
Use that log if the HailoRT output layout needs adjustment on-device.
