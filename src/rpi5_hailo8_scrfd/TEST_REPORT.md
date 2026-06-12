# SCRFD Test Report

Status: pending real-device validation on Raspberry Pi 5 + Hailo-8.

## Scope

This module was scaffolded for SCRFD face detection and includes:

- `model/scrfd_500m.hef`
- `model/scrfd_2.5g.hef`
- `model/scrfd_10g.hef`
- `docker/hailo8/scrfd.dockerfile`
- `POST /api/models/scrfd/predict`
- SCRFD raw score / bbox / landmark post-processing in `web_detection.py`

## Host Checklist

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0
```

Confirm the host `hailo-all`, firmware, mounted `libhailort.so`, and the wheel in
`hailort-packages/` use matching HailoRT major.minor versions.

## Build

```bash
cd src/rpi5_hailo8_scrfd

sudo docker build -f ../../docker/hailo8/scrfd.dockerfile \
    -t r20-hailo8-scrfd:latest .
```

## Demo Video Validation

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest
```

Expected:

- Web UI opens at `http://<Pi5_IP>:8000`
- face boxes are drawn on the preview
- 5-point landmarks are drawn as small yellow points when landmark outputs are present
- no HailoRT runtime errors in the container log

## API Validation

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/scrfd/predict -F "file=@test.jpg"
```

Expected JSON shape:

```json
{
  "success": true,
  "predictions": [
    {
      "class": "face",
      "confidence": 0.9,
      "box": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
      "landmarks": [{"x": 0, "y": 0}]
    }
  ]
}
```

## Model Switching

```bash
python web_detection.py --model_path model/scrfd_2.5g.hef --video_path video/test.mp4
python web_detection.py --model_path model/scrfd_10g.hef --video_path video/test.mp4
```

Record FPS, detection quality, and any output-shape mismatch after testing each
model on the Pi.
