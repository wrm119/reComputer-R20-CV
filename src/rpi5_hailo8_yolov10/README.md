# YOLOv11 on Raspberry Pi 5 + Hailo-8

This is the YOLOv11 object detection module for **Raspberry Pi 5 + Hailo-8**
(reComputer R20 series). It's designed as a **template**: copy this directory and
swap the `.hef` to adapt to other Hailo Model Zoo models (yolov11n/s/m, yolov8,
yolov5, yolov11-seg, etc.).

Features: real-time Web preview (MJPEG with detection overlay), REST API
compatible with Ultralytics Cloud API conventions, and offline batch video
analysis with libx264 ultrafast encoding.

---

## 1. Host preparation (Raspberry Pi 5)

The Hailo-8 PCIe driver, firmware and userspace tools must be installed on the host
**before** the container can talk to the chip.

```bash
# Raspberry Pi OS Bookworm (recommended)
sudo apt update
sudo apt install hailo-all
sudo reboot

# Verify the chip is detected
hailortcli fw-control identify
ls /dev/hailo0
```

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

`hailortcli fw-control identify` should print board info and a firmware version.
Remember the firmware version — your container's `hailort` wheel **must match it**.

---

## 2. Asset preparation

### 2.1 Download the model

Grab a pre-compiled `yolov11n.hef` from the
[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) and drop it into
`model/`:

```bash
cd src/rpi5_hailo8_yolov11/model
# Example path — check the Model Zoo for the version matching your HailoRT
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<version>/hailo8/yolov11n.hef
```

The `yolov11n.hef` shipped by the Model Zoo has the **NMS post-process layer baked
in**, which is what [web_detection.py](web_detection.py)'s `post_process_hailo()`
expects. If you compile your own `.hef` without that layer, the output will be the
3-branch raw tensor pair and you'll need to re-add DFL + box decoding.

### 2.2 Download the matching HailoRT Python wheel

Get a `hailort-<version>-cp311-cp311-linux_aarch64.whl` whose major.minor version
matches the host driver version reported by `hailortcli fw-control identify`. The
wheel comes from the Hailo Developer Zone (registration required).

Drop it into `hailort-packages/`:

```bash
cd src/rpi5_hailo8_yolov11/hailort-packages
# Example
ls hailort-4.23.0-cp311-cp311-linux_aarch64.whl
```

---

## 3. Run via Docker (recommended)

```bash
cd src/rpi5_hailo8_yolov11
sudo docker build -f ../../docker/hailo8/yolov11.dockerfile -t rpi5-hailo8-yolov11:latest .

# IMPORTANT: bind-mount the host's libhailort.so.<X.Y.Z>. The wheel installed
# inside the image only ships Python bindings; the native library must come
# from the host's `hailo-all` package and its major.minor MUST match the wheel.
# Find the exact path with: sudo find /usr -name "libhailort.so*"
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov11:latest
```

The container starts the Web preview on `http://<Pi5_IP>:8000` with the bundled
`video/test.mp4` looping.

### Camera mode

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov11:latest \
    python web_detection.py --model_path model/yolov11n.hef --camera_id 0
```

### Custom classes

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -v $(pwd)/class_config.txt:/app/class_config.txt \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov11:latest \
    python web_detection.py --model_path model/yolov11n.hef --video_path video/test.mp4 --class_path class_config.txt
```

`class_config.txt` is comma-separated, double-quoted names:
`"person", "bicycle", "car"`

---

## 4. Run without Docker

```bash
cd src/rpi5_hailo8_yolov11
pip install -r requirements.txt
pip install hailort-packages/hailort-*.whl

python web_detection.py --model_path model/yolov11n.hef --camera_id 0
# or
python web_detection.py --model_path model/yolov11n.hef --video_path video/test.mp4
```

---

## 5. REST API

Highlights (full endpoint list in the project root [README.md](../../README.md)):

- `POST /api/models/yolov11/predict` — single-shot inference on uploaded image, video
  frame, or current camera frame
- `GET  /api/video_feed` — MJPEG stream with detection boxes overlaid
- `GET  /POST /api/config` — read/update `obj_thresh` / `nms_thresh`
- `POST /api/video/upload`, `POST /api/video/analyze`, `GET /api/video/status`,
  `GET /api/video/download/{filename}` — local video batch analysis

`nms_thresh` is kept in the API for compatibility, but with the Model Zoo
`yolov11n.hef` NMS is already done on-chip, so this slider acts only as an
additional confidence-level filter knob. `obj_thresh` always filters detections
client-side after the chip returns them.

---

## 6. CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--model_path` | Path to `.hef` model | required |
| `--camera_id` | `/dev/videoN` index. `-1` = web-only mode | `0` |
| `--video_path` | Path to video file (overrides `--camera_id`) | none |
| `--class_path` | Path to `class_config.txt` for custom class names | none (COCO 80) |
| `--host` | Web server host | `0.0.0.0` |
| `--port` | Web server port | `8000` |
| `--preview_width` | MJPEG preview resize width (0 = native) | `1280` |
| `--preview_height` | MJPEG preview resize height (0 = native) | `720` |
| `--jpeg_quality` | MJPEG preview JPEG quality 1-100 | `80` |
| `--cam_width` | Requested USB camera width | `1280` |
| `--cam_height` | Requested USB camera height | `720` |
| `--target_fps` | Cap live preview inference rate (0 = uncapped) | `30` |

Tuning hints:
- Slow WiFi? Lower `--preview_width/height` further (e.g. 640x360) or drop
  `--jpeg_quality` to 60. Each step roughly halves the bytes per frame.
- Wired/local LAN? Set `--preview_width 0` to disable the resize and stream
  the native resolution.
- The MJPEG endpoint pushes the latest frame on a condvar, so a slow client
  never causes stale-frame pileup — the browser always sees something close
  to "now."
- While `/api/video/analyze` is running, the live preview automatically drops
  to 1 fps to free Hailo/CPU for the offline analysis (~2x speedup on 4K
  source). It resumes full rate when the analysis finishes.

---

## 7. Adapting to other models (yolov8, yolov5, yolov11-seg, etc.)

Use this directory as the template:

1. Copy the whole `src/rpi5_hailo8_yolov11/` folder, rename to e.g.
   `rpi5_hailo8_yolov5/`.
2. Download the matching `.hef` from the Model Zoo and drop in `model/`.
3. If the new model uses **the same output layout** (built-in NMS, `(1, num_classes,
   max_dets, 5)`), nothing else needs to change.
4. For seg / pose / obb / models without NMS layer, rewrite `post_process_hailo()`
   to handle the actual output tensors. The Model Zoo's per-model README documents
   the exact output spec.
5. Add a matching `docker/hailo8/<model>.dockerfile`.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Failed to open /dev/hailo0` inside container | Missing `--device /dev/hailo0:/dev/hailo0` |
| `libhailort.so.<X.Y.Z>: cannot open shared object file` | Missing the `-v /usr/lib/libhailort.so.<X.Y.Z>:...:ro` bind-mount. The wheel only contains Python bindings — the `.so` must come from the host. |
| `HailoRT firmware version mismatch` | Host driver and container wheel are different major.minor versions |
| Detection coordinates are off | The `.hef` you're using has a different input size — set `IMG_SIZE` in `web_detection.py` to match `hef.get_input_vstream_infos()[0].shape[:2]` |
| Single-digit FPS | You're probably rebuilding `InferVStreams` per frame — keep `HailoInfer` long-lived (default behavior of [py_utils/hailo_executor.py](py_utils/hailo_executor.py)) |
| `output.dtype == object` branch never hits / always hits | HailoRT version differs from what was tested; both branches do the same thing, so detections still work |

---

## 9. Validation & performance

See [TEST_REPORT.md](TEST_REPORT.md) for the full validation log:

- All endpoints exercised end-to-end on Pi 5 + Hailo-8 with a 4K source
- Hailo inference: 8.5 ms/frame (yolov11n.hef)
- LAN MJPEG: ~18 fps after V2 thread-split + 720p downscale (90× over V1)
- Offline analysis: 40 s for 394 frames of 4K (3× over V1, ffmpeg libx264 ultrafast)
- Pi 5 specific note: no `h264_v4l2m2m` HW encoder (Pi 4 only) — software encode required
