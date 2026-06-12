import argparse
import base64
import os
import queue
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e}), inference will fail", flush=True)


IMG_SIZE = (224, 224)  # (width, height); overwritten from HEF at startup.
ATTR_THRESH = 0.70
_ATTR_OUTPUT_LOGGED = False

PETA_LABELS = (
    "Age16-30",
    "Age31-45",
    "Age46-60",
    "AgeAbove61",
    "Backpack",
    "CarryingOther",
    "Casual lower",
    "Casual upper",
    "Formal lower",
    "Formal upper",
    "Hat",
    "Jacket",
    "Jeans",
    "Leather shoes",
    "Logo",
    "Long hair",
    "Male",
    "Messenger bag",
    "Muffler",
    "No accesory",
    "No carrying",
    "Plaid",
    "Plastic bag",
    "Sandals",
    "Shoes",
    "Shorts",
    "Short sleeve",
    "Skirt",
    "Sneaker",
    "Stripes",
    "Sunglasses",
    "Trousers",
    "T-shirt",
    "UpperOther",
    "V-Neck",
)

PETA_FILTERED = (
    "Age < 30",
    "Age 31-45",
    "Age 46-60",
    "Age 60+",
    "",
    "",
    "",
    "",
    "",
    "",
    "Hat",
    "",
    "",
    "",
    "Logo",
    "Long hair",
    "Male",
    "",
    "Muffler",
    "",
    "",
    "",
    "Plastic bag",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "Sunglasses",
    "",
    "",
    "",
    "",
)


class AttributeConfig:
    def __init__(self):
        self.attr_thresh = ATTR_THRESH
        self.lock = threading.Lock()

    def update(self, attr_thresh):
        with self.lock:
            self.attr_thresh = float(attr_thresh)

    def get(self):
        with self.lock:
            return self.attr_thresh


class CameraStream:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()


class LatestFrameQueue:
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)

    def put(self, frame):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(frame)

    def get(self):
        return self.queue.get()


app = FastAPI(title="reComputer Person Attribute ResNet (RPi5 + Hailo-8)")
det_config = AttributeConfig()
stop_event = threading.Event()
_global_model = None
_global_source = None


def _sigmoid(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_frame(frame):
    resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _select_output(outputs):
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        if not outputs:
            return None
        for name, value in outputs.items():
            arr = np.asarray(value)
            if arr.size >= len(PETA_LABELS):
                return arr, name
        name = next(iter(outputs))
        return np.asarray(outputs[name]), name
    return np.asarray(outputs), "output"


def post_process_hailo(outputs, threshold):
    global _ATTR_OUTPUT_LOGGED
    selected = _select_output(outputs)
    if selected is None:
        return []

    raw, output_name = selected
    raw_shape = raw.shape
    scores = raw.astype(np.float32).reshape(-1)
    if scores.size < len(PETA_LABELS):
        raise ValueError(f"Person attribute output too small: shape={raw_shape}, size={scores.size}")
    scores = scores[:len(PETA_LABELS)]

    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size and (float(finite_scores.min()) < 0.0 or float(finite_scores.max()) > 1.0):
        scores = _sigmoid(scores)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

    attributes = []
    for idx, (label, filtered_label, score) in enumerate(zip(PETA_LABELS, PETA_FILTERED, scores)):
        confidence = float(score)
        if filtered_label and confidence > threshold:
            attributes.append({
                "id": idx,
                "class": filtered_label,
                "raw_class": label,
                "confidence": confidence,
            })
        elif filtered_label == "Male" and confidence <= threshold:
            attributes.append({
                "id": idx,
                "class": "Female",
                "raw_class": label,
                "confidence": float(1.0 - confidence),
            })

    attributes.sort(key=lambda item: item["confidence"], reverse=True)

    if not _ATTR_OUTPUT_LOGGED:
        stats = f"min={float(scores.min()):.4f}, max={float(scores.max()):.4f}, mean={float(scores.mean()):.4f}"
        print(
            f"[PersonAttrResNet] output={output_name}, raw shape={raw_shape}, decoded attrs={len(attributes)}, {stats}",
            flush=True,
        )
        _ATTR_OUTPUT_LOGGED = True

    return attributes


def draw_attributes(frame, attributes):
    panel_w = min(frame.shape[1] - 20, 440)
    rows = max(1, min(len(attributes), 8))
    panel_h = 42 + rows * 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (16 + panel_w, 16 + panel_h), (15, 20, 24), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, "Person Attribute", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 255, 80), 2)

    if not attributes:
        cv2.putText(frame, "No attribute above threshold", (30, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2)
        return

    for i, attr in enumerate(attributes[:8]):
        y = 76 + i * 30
        text = f"{attr['class']}: {attr['confidence']:.2f}"
        cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)


def infer_frame(frame, model, threshold):
    input_img = preprocess_frame(frame)
    outputs = model.run(input_img)
    attrs = post_process_hailo(outputs, threshold)
    draw_attributes(frame, attrs)
    return attrs


def frame_generator(model, source):
    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    stream = CameraStream(source).start()
    try:
        while not stop_event.is_set():
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            threshold = det_config.get()
            if model is not None:
                try:
                    infer_frame(frame, model, threshold)
                except Exception as e:
                    cv2.putText(frame, f"Inference error: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now
            cv2.putText(frame, f"Hailo FPS: {fps:.1f}", (20, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
    finally:
        stream.stop()


@app.get("/")
async def index():
    return Response(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Person Attribute ResNet</title>
  <style>
    body { margin: 0; background: #171a1f; color: #e8edf2; font-family: Arial, sans-serif; }
    main { min-height: 100vh; display: grid; place-items: center; padding: 24px; box-sizing: border-box; }
    .wrap { width: min(96vw, 1280px); }
    .bar { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
    label { color: #c7d0da; font-size: 14px; }
    input[type=range] { width: 220px; }
    img { width: 100%; border: 6px solid #3a3d42; border-radius: 8px; background: #0e1013; display: block; }
  </style>
</head>
<body>
  <main>
    <div class="wrap">
      <div class="bar">
        <strong>Person Attribute ResNet</strong>
        <label>Threshold <span id="val">0.70</span></label>
        <input id="threshold" type="range" min="0.05" max="0.95" step="0.01" value="0.70">
      </div>
      <img src="/video_feed" alt="Person attribute stream">
    </div>
  </main>
  <script>
    const slider = document.getElementById('threshold');
    const val = document.getElementById('val');
    async function update() {
      val.textContent = Number(slider.value).toFixed(2);
      await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({attr_thresh: Number(slider.value)})
      });
    }
    slider.addEventListener('input', update);
  </script>
</body>
</html>
        """,
        media_type="text/html",
    )


@app.get("/video_feed")
async def video_feed():
    if _global_model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")
    return StreamingResponse(frame_generator(_global_model, _global_source), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/config")
async def get_config():
    return {"attr_thresh": det_config.get()}


@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("attr_thresh", ATTR_THRESH))
    return {"status": "success", "attr_thresh": det_config.get()}


@app.post("/api/models/person_attr_resnet/predict")
async def predict(file: UploadFile = File(...), threshold: Optional[float] = None):
    if _global_model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    data = await file.read()
    img_array = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    work = frame.copy()
    attr_thresh = det_config.get() if threshold is None else float(threshold)
    attributes = infer_frame(work, _global_model, attr_thresh)

    ok, buffer = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")

    return {
        "model": "person_attr_resnet",
        "predictions": attributes,
        "attributes": attributes,
        "threshold": attr_thresh,
        "image": base64.b64encode(buffer).decode("utf-8"),
    }


def parse_source(args):
    if args.video_path and os.path.exists(args.video_path):
        return args.video_path
    return args.camera_id


def main():
    parser = argparse.ArgumentParser(description="Person Attribute ResNet on RPi5 + Hailo-8")
    parser.add_argument("--model_path", default="model/person_attr_resnet_v1_18.hef")
    parser.add_argument("--video_path", default="video/test.mp4")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threshold", type=float, default=ATTR_THRESH)
    args = parser.parse_args()

    global _global_model, _global_source, IMG_SIZE

    det_config.update(args.threshold)
    _global_source = parse_source(args)

    if not HAILO_AVAILABLE:
        raise RuntimeError("HailoRT is not available. Install the matching hailort wheel in the Docker image.")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"HEF model not found: {args.model_path}")

    _global_model = HailoInfer(args.model_path)
    IMG_SIZE = (int(_global_model.input_w), int(_global_model.input_h))
    print(f"[PersonAttrResNet] model={args.model_path}, input={IMG_SIZE[0]}x{IMG_SIZE[1]}x3", flush=True)
    print(f"[PersonAttrResNet] source={_global_source}, threshold={det_config.get():.2f}", flush=True)

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        stop_event.set()
        if _global_model is not None:
            _global_model.release()


if __name__ == "__main__":
    main()
