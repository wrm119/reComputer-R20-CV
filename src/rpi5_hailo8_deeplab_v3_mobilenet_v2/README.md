# unet v3 MobileNet v2 on Raspberry Pi 5 + Hailo-8

This is the unet v3 MobileNet v2 **semantic segmentation** module for
**Raspberry Pi 5 + Hailo-8** (reComputer R20 series). It started as a fork of
the YOLOv5 template (the REST routes still carry the `/api/models/yolov5/...`
path for backward compatibility) and has been retargeted to a 513×513×3
segmentation model.

Features: real-time Web preview (MJPEG with per-pixel class mask overlaid on
the original frame), REST API compatible with the existing reComputer-CV
conventions, and offline batch video analysis with libx264 ultrafast encoding.

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

Grab a pre-compiled `unet_mobilenet_v2.hef` from the
[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) and drop it into
`model/`:

```bash
cd src/rpi5_hailo8_unet_mobilenet_v2/model
# Example path — check the Model Zoo for the version matching your HailoRT
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<version>/hailo8/unet_mobilenet_v2.hef
```

The Model Zoo ships the model trained on **PASCAL VOC (21 classes,
including `background`)** with input shape `513×513×3` and output shape
`(1, 513, 513, 21)`. This is what [web_detection.py](web_detection.py)'s
`post_process_hailo()` expects. If you recompile from a different training set
(Cityscapes, ADE20K, etc.) you must also pass `--class_path` so the per-pixel
class indices map to the right names.

### 2.2 Download the matching HailoRT Python wheel

Get a `hailort-<version>-cp311-cp311-linux_aarch64.whl` whose major.minor version
matches the host driver version reported by `hailortcli fw-control identify`. The
wheel comes from the Hailo Developer Zone (registration required).

Drop it into `hailort-packages/`:

```bash
cd src/rpi5_hailo8_unet_mobilenet_v2/hailort-packages
# Example
ls hailort-4.23.0-cp311-cp311-linux_aarch64.whl
```

---

## 3. Run via Docker (recommended)

```bash
cd src/rpi5_hailo8_unet_mobilenet_v2
sudo docker build -f ../../docker/hailo8/unet_mobilenet_v2.dockerfile -t rpi5-hailo8-unet:latest .

# IMPORTANT: bind-mount the host's libhailort.so.<X.Y.Z>. The wheel installed
# inside the image only ships Python bindings; the native library must come
# from the host's `hailo-all` package and its major.minor MUST match the wheel.
# Find the exact path with: sudo find /usr -name "libhailort.so*"
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-unet:latest
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
    rpi5-hailo8-unet:latest \
    python web_detection.py --model_path model/unet_mobilenet_v2.hef --camera_id 0
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
    rpi5-hailo8-unet:latest \
    python web_detection.py --model_path model/unet_mobilenet_v2.hef --video_path video/test.mp4 --class_path class_config.txt
```

`class_config.txt` is comma-separated, double-quoted names — the **first** entry
must be the background class (its mask region is left uncolored on the preview):

```
"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", ...
```

---

## 4. Run without Docker

```bash
cd src/rpi5_hailo8_unet_mobilenet_v2
pip install -r requirements.txt
pip install hailort-packages/hailort-*.whl

python web_detection.py --model_path model/unet_mobilenet_v2.hef --camera_id 0
# or
python web_detection.py --model_path model/unet_mobilenet_v2.hef --video_path video/test.mp4
```

---

## 5. REST API

Highlights (full endpoint list in the project root [README.md](../../README.md)):

- `POST /api/models/unet/predict` — single-shot inference on uploaded image, video
  frame, or current camera frame. **Path retained for backward compatibility**;
  the response `predictions` field is now a list of
  `{class, confidence, pixels}` entries — one per non-background class found in
  the segmentation mask. `confidence` is the fraction of mask pixels covered by
  that class (0–1); `pixels` is the raw pixel count at network resolution
  (513×513).
- `GET  /api/video_feed` — MJPEG stream with the segmentation mask alpha-blended
  over the original frame.
- `GET / POST /api/config` — read/update `obj_thresh` / `nms_thresh`. Both
  fields are kept in the API for backward compatibility but are not used by the
  segmentation pipeline.
- `POST /api/video/upload`, `POST /api/video/analyze`, `GET /api/video/status`,
  `GET /api/video/download/{filename}` — local video batch analysis.

---

## 6. CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--model_path` | Path to `.hef` model | required |
| `--camera_id` | `/dev/videoN` index. `-1` = web-only mode | `0` |
| `--video_path` | Path to video file (overrides `--camera_id`) | none |
| `--class_path` | Path to `class_config.txt` for custom class names | none (PASCAL VOC 21) |
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
  to 1 fps to free Hailo/CPU for the offline analysis. It resumes full rate
  when the analysis finishes.

---

## 7. Adapting to other segmentation models

Use this directory as the template:

1. Copy the whole `src/rpi5_hailo8_unet_mobilenet_v2/` folder and rename it.
2. Download the matching `.hef` from the Model Zoo and drop in `model/`.
3. If the new model uses the same `(1, H, W, num_classes)` per-pixel softmax
   output, only `IMG_SIZE` in `web_detection.py` and the default class list
   need adjusting.
4. If channels come back as `(1, num_classes, H, W)`, `post_process_hailo()`
   already detects this and transposes — no change needed.
5. For instance segmentation, panoptic, or models with a different output
   schema, rewrite `post_process_hailo()` to return a 2D class-index mask of
   shape `(input_h, input_w)`.
6. Add a matching `docker/hailo8/<model>.dockerfile`.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Failed to open /dev/hailo0` inside container | Missing `--device /dev/hailo0:/dev/hailo0` |
| `libhailort.so.<X.Y.Z>: cannot open shared object file` | Missing the `-v /usr/lib/libhailort.so.<X.Y.Z>:...:ro` bind-mount. The wheel only contains Python bindings — the `.so` must come from the host. |
| `HailoRT firmware version mismatch` | Host driver and container wheel are different major.minor versions |
| Mask appears shifted/clipped on the preview | The `.hef` input size doesn't match `IMG_SIZE` — set it to `hef.get_input_vstream_infos()[0].shape[:2]` (default `(513, 513)`) |
| Wrong colors / wrong class names on overlay | Your `.hef` was trained on a non-VOC dataset. Pass `--class_path class_config.txt` listing the actual classes in index order, with `background` first. |
| Single-digit FPS | You're probably rebuilding `InferVStreams` per frame — keep `HailoInfer` long-lived (default behavior of [py_utils/hailo_executor.py](py_utils/hailo_executor.py)) |

---

## 9. Notes on performance

- Inference cost is dominated by the 513×513 forward pass; the segmentation
  decode is a single `np.argmax` over the channel axis (~negligible vs. the
  Hailo call).
- The mask is **nearest-neighbor resized back to the original frame size**
  before alpha blending, so 4K input still renders crisp class boundaries.
- The inference / encode thread split is the same as the YOLOv5 template:
  inference pushes annotated frames into a condvar-backed buffer, the encode
  thread JPEG-encodes the latest one. Slow clients can never back-pressure
  inference.

There is currently no published validation report for this module; the
performance characteristics of the inference / preview / offline-analysis
pipeline carry over from the YOLOv5 template since the threading model is
unchanged.
