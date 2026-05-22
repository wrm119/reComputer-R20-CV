# reComputer-R20-CV

[English] | [中文](./README_zh.md)

Industrial-grade computer vision reference for **Raspberry Pi 5 + Hailo-8**
(reComputer R20 series). YOLOv8 object detection with real-time MJPEG preview,
REST API, and offline batch video analysis — designed around Pi 5's actual
hardware (no `h264_v4l2m2m` HW encoder, modest LAN bandwidth, PCIe-attached
Hailo-8 accelerator).

The single module under `src/` is built as a **template**: copy it and swap the
`.hef` to retarget to other Hailo Model Zoo models (yolov8s/m, yolov5,
yolov8-seg, etc.).

---

## Hardware platform

| | |
|---|---|
| Board | Raspberry Pi 5 (reComputer R20 series carrier) |
| Accelerator | Hailo-8 M.2 (PCIe), device node `/dev/hailo0` |
| OS | Raspberry Pi OS Bookworm, kernel 6.12+ aarch64 |
| Host drivers | `hailo-all` apt package (provides PCIe driver, firmware, `libhailort.so`) |
| HailoRT | 4.23.x validated — host driver / firmware / container wheel **must share major.minor** |

---

## Quick start (one-command, pre-built image)

The published image already contains the source code, HailoRT wheel, ffmpeg, and
the three Model Zoo `.hef` weights (`yolov8n/s/m`). You only need a working Hailo
toolchain on the host.

### 1. Host prep (one-time, on the Pi)

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

# After reboot, confirm the chip and note the firmware version
hailortcli fw-control identify     # should report 4.23.0
ls /dev/hailo0
```

### 2. Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/wrm119/recomputer-r20-cv:latest
```

Docker will pull the image on first run (~1.8 GB). The container then loops the
bundled `video/test.mp4` and serves the Web UI on port `8000` — open
`http://<Pi5_IP>:8000` in a browser.

> **Why the `libhailort.so` bind-mount?** The image ships only the Python
> bindings; the native library has to come from the host's `hailo-all` package.
> If your firmware version isn't `4.23.0`, replace both `4.23.0` references with
> the version printed by `hailortcli fw-control identify` (and rebuild the image
> from source against a matching wheel if the major.minor differs).

### USB camera mode

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/wrm119/recomputer-r20-cv:latest \
    python web_detection.py --model_path model/yolov8n.hef --camera_id 0
```

---

## Directory layout

```text
reComputer-R20-CV/
├── docker/hailo8/
│   └── yolov8.dockerfile           # Image: python:3.11-slim arm64 + ffmpeg + HailoRT wheel
└── src/rpi5_hailo8_yolov8/
    ├── web_detection.py            # FastAPI + inference/encode threading pipeline
    ├── py_utils/
    │   ├── hailo_executor.py       # HailoRT wrapper, long-lived InferVStreams
    │   └── coco_utils.py           # Letterbox + box coordinate restoration
    ├── model/                      # `.hef` weights (bundled in the image)
    ├── hailort-packages/           # HailoRT wheel (bundled in the image)
    ├── video/test.mp4              # Bundled demo source
    ├── requirements.txt
    ├── README.md / README_zh.md    # Module deep dive: deployment, CLI, troubleshooting
    └── TEST_REPORT.md              # V1→V2 perf analysis, validation log
```

---

## Build from source (for customization)

When you need to change the code, swap a different `.hef`, or rebuild against a
different HailoRT version:

```bash
git clone https://github.com/wrm119/reComputer-R20-CV.git
cd reComputer-R20-CV/src/rpi5_hailo8_yolov8

# Replace assets if needed (already bundled — only swap to customize)
# ls model/                   # .hef files
# ls hailort-packages/        # hailort wheel

sudo docker build -f ../../docker/hailo8/yolov8.dockerfile \
    -t r20-hailo8-yolov8:latest .

# Same run command, with the local tag instead of ghcr.io
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8:latest
```

---

## REST API

All endpoints listen on port `8000` of the container; with `--net=host` they're
reachable at `http://<Pi5_IP>:8000`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/models/yolov8/predict` | POST | One-shot inference on uploaded image, specific video frame, or current camera frame |
| `/api/video_feed` | GET | MJPEG live stream with boxes overlaid (embed in an `<img>`) |
| `/api/config` | GET / POST | Read or update `obj_thresh` / `nms_thresh` |
| `/api/video/upload` | POST | Upload a video for batch analysis |
| `/api/video/analyze` | POST | Start an offline analysis job (form-data `filename=...`) |
| `/api/video/status` | GET | Poll job progress |
| `/api/video/list` | GET | List uploaded sources and finished outputs |
| `/api/video/download/{filename}` | GET | Download an annotated output |

### Inference examples

```bash
# Image upload
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict -F "file=@cat.jpg"

# Specific frame of an uploaded video (timestamp in seconds)
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict \
    -F "video=@test.mp4" -F "timestamp=5.5"

# Current camera frame
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict -F "realtime=true"

# Per-call threshold override
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict \
    -F "file=@cat.jpg" -F "conf=0.5" -F "iou=0.4"
```

Response:

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "car",
      "confidence": 0.787,
      "box": { "x1": 2108, "y1": 1483, "x2": 2291, "y2": 1651 }
    }
  ],
  "image": { "width": 3840, "height": 2160 }
}
```

Embed the live stream in any HTML page:

```html
<img src="http://<Pi5_IP>:8000/api/video_feed">
```

### Dynamic threshold update

```bash
# Read current
curl http://<Pi5_IP>:8000/api/config
# {"obj_thresh":0.25,"nms_thresh":0.45}

# Update (either field is optional)
curl -X POST http://<Pi5_IP>:8000/api/config \
     -H "Content-Type: application/json" \
     -d '{"obj_thresh":0.4}'
```

> `nms_thresh` is kept for API compatibility, but the Model Zoo `yolov8n.hef`
> performs NMS on-chip — the slider only acts as an extra confidence filter.

---

## Adapting to other models

Treat [src/rpi5_hailo8_yolov8/](src/rpi5_hailo8_yolov8/) as the template:

1. Copy the whole directory, rename (e.g. `rpi5_hailo8_yolov8_seg/`).
2. Drop the new `.hef` into `model/`.
3. If the model uses the **same output layout** (NMS on-chip,
   `(1, num_classes, max_dets, 5)`) — nothing else changes.
4. For seg / pose / obb / non-NMS models, rewrite `post_process_hailo()` in
   `web_detection.py` against the actual tensor spec. Hailo Model Zoo's per-model
   README documents the output layout.
5. Add a matching `docker/hailo8/<model>.dockerfile`.

Walkthrough:
[src/rpi5_hailo8_yolov8/README.md § 7](src/rpi5_hailo8_yolov8/README.md#7-adapting-to-other-models).

---

## Documentation

- [src/rpi5_hailo8_yolov8/README.md](src/rpi5_hailo8_yolov8/README.md) — module-level
  deployment, CLI arguments, troubleshooting (English)
- [src/rpi5_hailo8_yolov8/README_zh.md](src/rpi5_hailo8_yolov8/README_zh.md) — 同上中文版
- [src/rpi5_hailo8_yolov8/TEST_REPORT.md](src/rpi5_hailo8_yolov8/TEST_REPORT.md) — V1→V2 performance analysis, end-to-end validation log
