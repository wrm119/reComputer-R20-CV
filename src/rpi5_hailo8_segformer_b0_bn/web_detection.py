import os
import sys
import cv2
import argparse
import time
import queue
import subprocess
import numpy as np
import threading
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import shutil
from typing import Optional, List

from py_utils.coco_utils import COCO_test_helper

stop_event = threading.Event()

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e}), inference will fail")

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (1024, 512)  # (width, height) must match the .hef input shape

DEFAULT_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
)

# Cityscapes trainId palette in BGR order because OpenCV images are BGR.
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # road
    [232, 35, 244],   # sidewalk
    [70, 70, 70],     # building
    [156, 102, 102],  # wall
    [153, 153, 190],  # fence
    [153, 153, 153],  # pole
    [30, 170, 250],   # traffic light
    [0, 220, 220],    # traffic sign
    [35, 142, 107],   # vegetation
    [152, 251, 152],  # terrain
    [180, 130, 70],   # sky
    [60, 20, 220],    # person
    [0, 0, 255],      # rider
    [142, 0, 0],      # car
    [70, 0, 0],       # truck
    [100, 60, 0],     # bus
    [100, 80, 0],     # train
    [230, 0, 0],      # motorcycle
    [32, 11, 119],    # bicycle
], dtype=np.uint8)

CLASSES = DEFAULT_CLASSES
_SEG_OUTPUT_LOGGED = False

def load_classes(path):
    global CLASSES
    if not path or not os.path.exists(path):
        CLASSES = DEFAULT_CLASSES
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            import re
            items = re.findall(r'"([^"]*)"', content)
            if items:
                CLASSES = tuple(items)
                print(f"Successfully loaded {len(CLASSES)} classes from {path}")
            else:
                items = [item.strip().strip('"') for item in content.split(',') if item.strip()]
                if items:
                    CLASSES = tuple(items)
                    print(f"Loaded {len(CLASSES)} classes from {path} (fallback parsing)")
                else:
                    print(f"Warning: No classes found in {path}, using default Cityscapes classes")
                    CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using default Cityscapes classes")
        CLASSES = DEFAULT_CLASSES

class DetectionConfig:
    def __init__(self):
        self.obj_thresh = 0.25
        self.nms_thresh = 0.45
        self.lock = threading.Lock()

    def update(self, obj_thresh, nms_thresh):
        with self.lock:
            self.obj_thresh = obj_thresh
            self.nms_thresh = nms_thresh

    def get(self):
        with self.lock:
            return self.obj_thresh, self.nms_thresh

det_config = DetectionConfig()

UPLOAD_DIR = "workspace/uploads"
OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VideoAnalyzer:
    def __init__(self, model=None, co_helper=None):
        self.model = model
        self.co_helper = co_helper
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None

    def set_engine(self, model, co_helper):
        self.model = model
        self.co_helper = co_helper

    def start_analysis(self, input_path, output_path):
        if self.is_processing:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_video, args=(input_path, output_path))
        self._thread.daemon = True
        self._thread.start()
        return True

    @staticmethod
    def _open_writer(output_path, width, height, fps):
        """Try ffmpeg subprocess with libx264 ultrafast first 鈥?on Pi 5 this
        is ~5x faster than cv2's mp4v at 4K (40ms vs 150ms per frame).
        Falls back to cv2 mp4v if ffmpeg is not installed (e.g., non-Docker run).

        Returns (writer, kind) where kind is 'ffmpeg' or 'mp4v'. The caller
        uses kind to decide whether to write via proc.stdin.write or out.write."""
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', f'{fps}',
            '-i', '-',
            # ultrafast = simplest motion search; threads=0 lets x264 use all 4 cores.
            # NOT using -tune zerolatency: it forces single-thread sliced threading,
            # which on Pi 5 cuts throughput in half. For offline encoding we want
            # frame-level threading instead.
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '0',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            print(f"[VideoAnalyzer] Using ffmpeg libx264 ultrafast", flush=True)
            return proc, 'ffmpeg'
        except FileNotFoundError:
            pass

        # Fallback: cv2 mp4v (works without ffmpeg binary, but slow at 4K)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"[VideoAnalyzer] Using cv2 mp4v (slower; install ffmpeg for 5x speedup)", flush=True)
            return out, 'mp4v'
        out.release()
        return None, None

    def _process_video(self, input_path, output_path):
        self.is_processing = True
        self.progress = 0
        self.error_msg = ""
        self.current_video = os.path.basename(input_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.error_msg = f"Error: Cannot open video {input_path}"
            self.is_processing = False
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            self.error_msg = "Error: Invalid total frames"
            self.is_processing = False
            cap.release()
            return

        out, kind = self._open_writer(output_path, width, height, fps)
        if out is None:
            self.error_msg = "Error: No usable video writer (ffmpeg + cv2 mp4v both failed)"
            self.is_processing = False
            cap.release()
            return

        # Single-threaded loop 鈥?empirical: on Pi 5 with 4K frames, the
        # producer/consumer pipeline experiment was 2-3x SLOWER than the
        # straight loop. Likely because 4K BGR frames (~24MB each) in a
        # Python queue trigger heavy GC, and libavcodec's mp4v encoder
        # doesn't play nicely with concurrent libav decoders on the same
        # process. The straight loop is the boring-but-fast choice here.
        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if self.model and self.co_helper:
                    processed_img, lb_info = preprocess_frame(frame, self.co_helper)
                    outputs = self.model.run(processed_img)

                    if outputs is not None:
                        # obj, nms = det_config.get()
                        mask = post_process_hailo(outputs, 0, 0, IMG_SIZE[1], IMG_SIZE[0])
                        if mask is not None:
                            draw_segmentation_mask(frame, mask, lb_info)
                if kind == 'ffmpeg':
                    out.stdin.write(frame.tobytes())
                else:
                    out.write(frame)
                frame_idx += 1
                self.progress = int((frame_idx / total_frames) * 100)

        except Exception as e:
            self.error_msg = f"Process error: {str(e)}"
        finally:
            cap.release()
            if kind == 'ffmpeg':
                try:
                    out.stdin.close()
                except Exception:
                    pass
                try:
                    out.wait(timeout=30)
                except Exception:
                    out.kill()
            else:
                out.release()
            self.is_processing = False
            if not self.error_msg:
                self.progress = 100

    def stop(self):
        self._stop_event.set()

video_analyzer = VideoAnalyzer()

app = FastAPI(title="reComputer Hailo-CV Web Preview (RPi5 + Hailo-8)")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get()
    return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", 0.25), config.get("nms_thresh", 0.45))
    return {"status": "success"}

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    uploads = os.listdir(UPLOAD_DIR)
    outputs = os.listdir(OUTPUT_DIR)
    return {"uploads": uploads, "outputs": outputs}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    name_base = os.path.splitext(filename)[0]
    output_filename = f"{name_base}_{width}x{height}_results.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    success = video_analyzer.start_analysis(input_path, output_path)
    if success:
        return {"status": "started", "output": output_filename}
    else:
        return {"status": "error", "message": "Already processing another video"}

@app.get("/api/video/status")
async def get_analysis_status():
    return {
        "is_processing": video_analyzer.is_processing,
        "progress": video_analyzer.progress,
        "current_video": video_analyzer.current_video,
        "error": video_analyzer.error_msg
    }

@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4', filename=filename)

_global_model = None
_global_co_helper = None

@app.post("/api/models/segformer/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
    realtime: Optional[bool] = Form(False),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None)
):
    if _global_model is None or _global_co_helper is None:
        return {"success": False, "message": "Model not initialized"}

    try:
        img = None
        source_info = ""

        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_info = "uploaded image"

        elif video:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await video.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                if timestamp is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                if ret:
                    img = frame
                    source_info = f"video frame at {timestamp if timestamp else 0}s"
                cap.release()
            os.unlink(tmp_path)

        if img is None:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"

        if img is None:
            return {"success": False, "message": "No valid input source found (image, video, or camera)"}

        h, w = img.shape[:2]

        input_img, lb_info = preprocess_frame(img, _global_co_helper)
        outputs = _global_model.run(input_img)

        mask = post_process_hailo(outputs, 0, 0, IMG_SIZE[1], IMG_SIZE[0])

        predictions = []
        if mask is not None:
            draw_segmentation_mask(img, mask, lb_info)

            # Summarize segmentation as per-class pixel coverage. The "confidence"
            # field carries the fraction of the network-input mask occupied by
            # each class so existing API consumers see a familiar shape.
            class_ids, counts = np.unique(mask, return_counts=True)
            total = float(mask.size)
            for cl, n in zip(class_ids, counts):
                if cl == 255 or cl >= len(CLASSES):
                    continue
                predictions.append({
                    "class": CLASSES[int(cl)],
                    "confidence": float(n) / total,
                    "pixels": int(n),
                })

        return {
            "success": True,
            "source": source_info,
            "predictions": predictions,
            "image": {
                "width": w,
                "height": h
            }
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

class FrameBuffer:
    """Latest-frame buffer with version tracking.

    Two stages separated by version counters:
      - annotated: 4K BGR frame from the inference thread, waiting to be encoded.
                   Encode thread blocks on wait_annotated().
      - jpeg:      downscaled preview JPEG bytes, ready to ship to browsers.
                   MJPEG stream consumers block on wait_jpeg().

    `raw` mirrors the latest annotated frame for the realtime predict API.
    Consumers wake on a condvar instead of polling, so a slow network can't
    cause stale-frame pileup.
    """
    def __init__(self):
        self.raw = None
        self.annotated = None
        self.annotated_version = 0
        self.jpeg = None
        self.jpeg_version = 0
        self.cond = threading.Condition()

    def push_annotated(self, frame):
        with self.cond:
            self.raw = frame
            self.annotated = frame
            self.annotated_version += 1
            self.cond.notify_all()

    def wait_annotated(self, last_version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.annotated_version > last_version, timeout=timeout)
            return self.annotated, self.annotated_version

    def push_jpeg(self, jpeg_bytes):
        with self.cond:
            self.jpeg = jpeg_bytes
            self.jpeg_version += 1
            self.cond.notify_all()

    def wait_jpeg(self, last_version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.jpeg_version > last_version, timeout=timeout)
            return self.jpeg, self.jpeg_version

    def get_raw_frame(self):
        with self.cond:
            return self.raw.copy() if self.raw is not None else None

frame_buffer = FrameBuffer()

class LatestFrameReader:
    """Continuously read a live camera and expose only the newest frame.

    USB/V4L2 camera buffers can queue old frames when inference is slower than
    capture. A single-threaded read/infer loop then shows stale frames even if
    the web stream itself has no queue. This reader drains the camera in the
    background and lets inference always consume the latest available frame.
    """
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.version = 0
        self._last_read_version = 0
        self._stopped = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._cond:
                self.frame = frame
                self.version += 1
                self._cond.notify_all()
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def read(self, timeout=1.0):
        with self._cond:
            self._cond.wait_for(
                lambda: self.version > self._last_read_version or self._stopped,
                timeout=timeout
            )
            if self.frame is None:
                return False, None
            self._last_read_version = self.version
            return True, self.frame.copy()

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=2)

@app.get("/api/video_feed")
async def video_feed():
    def generate():
        last_v = -1
        while True:
            jpeg, last_v = frame_buffer.wait_jpeg(last_v, timeout=1.0)
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    return Response(content="""
    <html>
      <head>
        <title>reComputer Hailo-CV Web Preview</title>
        <style>
          body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
          .container { max-width: 1200px; margin: 0 auto; }
          .video-box { margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden; background: #000; width: 100%; max-width: 800px; }
          .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; min-width: 400px; vertical-align: top; margin: 10px; }
          .control-group { margin-bottom: 15px; }
          .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
          .slider-container { display: flex; align-items: center; gap: 15px; }
          input[type=range] { flex-grow: 1; cursor: pointer; }
          .value-display { min-width: 50px; font-family: monospace; background: #444; padding: 2px 8px; border-radius: 4px; text-align: center; }
          h1 { color: #00e676; }
          .tab-container { margin-top: 30px; }
          .tabs { display: flex; justify-content: center; margin-bottom: 20px; border-bottom: 2px solid #333; }
          .tab { padding: 10px 30px; cursor: pointer; border-bottom: 3px solid transparent; transition: 0.3s; font-weight: bold; }
          .tab.active { border-bottom-color: #00e676; color: #00e676; }
          .tab-content { display: none; }
          .tab-content.active { display: block; }
          .video-analysis { text-align: left; background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 10px; }
          .btn { background: #00e676; color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; margin: 5px; }
          .btn:hover { background: #00c853; }
          .btn:disabled { background: #555; cursor: not-allowed; }
          .progress-container { width: 100%; background: #444; border-radius: 10px; margin: 15px 0; height: 20px; position: relative; overflow: hidden; }
          .progress-bar { height: 100%; background: #00e676; width: 0%; transition: 0.3s; }
          .progress-text { position: absolute; width: 100%; text-align: center; top: 0; left: 0; line-height: 20px; font-size: 12px; font-weight: bold; color: #fff; text-shadow: 1px 1px 2px #000; }
          table { width: 100%; border-collapse: collapse; margin-top: 15px; }
          th, td { text-align: left; padding: 10px; border-bottom: 1px solid #444; }
          th { color: #888; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>RPi5 + Hailo-8 Real-time Detection</h1>

          <div class="tabs">
            <div class="tab active" onclick="showTab('realtime')">Real-time Detection</div>
            <div class="tab" onclick="showTab('analysis')">Local Video Analysis</div>
          </div>

          <div id="realtime" class="tab-content active">
            <div class="video-box">
              <img id="streamImg" src="/api/video_feed" style="max-width: 100%; height: auto;">
            </div>

            <div class="controls">
              <div class="control-group">
                <label>Confidence Threshold</label>
                <div class="slider-container">
                  <input type="range" id="confSlider" min="0.01" max="1.0" step="0.01" value="0.25">
                  <span id="confValue" class="value-display">0.25</span>
                </div>
              </div>

              <div class="control-group">
                <label>IOU Threshold (applied on top of built-in NMS)</label>
                <div class="slider-container">
                  <input type="range" id="iouSlider" min="0.01" max="1.0" step="0.01" value="0.45">
                  <span id="iouValue" class="value-display">0.45</span>
                </div>
              </div>
            </div>
          </div>

          <div id="analysis" class="tab-content">
            <div class="video-analysis">
              <h3>Analyze Local Video</h3>
              <div class="control-group">
                <label>Upload New Video (.mp4)</label>
                <input type="file" id="videoUpload" accept=".mp4">
                <button class="btn" onclick="uploadVideo()">Upload</button>
              </div>

              <div id="processingArea" style="display: none;">
                <p id="statusText">Processing: <span id="currentFileName">-</span></p>
                <div class="progress-container">
                  <div id="progressBar" class="progress-bar"></div>
                  <div id="progressText" class="progress-text">0%</div>
                </div>
                <p id="errorText" style="color: #ff5252;"></p>
              </div>

              <div class="control-group">
                <label>File Management</label>
                <button class="btn" onclick="refreshFileList()">Refresh List</button>
                <table>
                  <thead>
                    <tr><th>File Name</th><th>Action</th></tr>
                  </thead>
                  <tbody id="fileTableBody">
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <p style="color: #888; margin-top: 20px;">Streaming via FastAPI + MJPEG | Port: 8000</p>
        </div>

        <script>
          function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');

            if (tabId === 'realtime') {
                document.getElementById('streamImg').src = '/api/video_feed';
            } else {
                document.getElementById('streamImg').src = '';
                refreshFileList();
            }
          }

          const confSlider = document.getElementById('confSlider');
          const iouSlider = document.getElementById('iouSlider');
          const confValue = document.getElementById('confValue');
          const iouValue = document.getElementById('iouValue');

          function updateConfig() {
            const obj_thresh = parseFloat(confSlider.value);
            const nms_thresh = parseFloat(iouSlider.value);
            confValue.innerText = obj_thresh.toFixed(2);
            iouValue.innerText = nms_thresh.toFixed(2);

            fetch('/api/config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ obj_thresh, nms_thresh })
            });
          }

          confSlider.oninput = updateConfig;
          iouSlider.oninput = updateConfig;

          fetch('/api/config').then(res => res.json()).then(data => {
            confSlider.value = data.obj_thresh;
            iouSlider.value = data.nms_thresh;
            confValue.innerText = data.obj_thresh.toFixed(2);
            iouValue.innerText = data.nms_thresh.toFixed(2);
          });

          async function uploadVideo() {
            const fileInput = document.getElementById('videoUpload');
            if (!fileInput.files[0]) return alert('Please select a file');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const btn = event.currentTarget;
            btn.disabled = true;
            btn.innerText = 'Uploading...';

            try {
                await fetch('/api/video/upload', { method: 'POST', body: formData });
                alert('Upload successful');
                refreshFileList();
            } catch (e) {
                alert('Upload failed');
            } finally {
                btn.disabled = false;
                btn.innerText = 'Upload';
            }
          }

          async function refreshFileList() {
            const res = await fetch('/api/video/list');
            const data = await res.json();
            const tbody = document.getElementById('fileTableBody');
            tbody.innerHTML = '';

            data.uploads.forEach(f => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${f} (Original)</td>
                    <td><button class="btn" onclick="analyzeVideo('${f}')">Analyze</button></td>
                `;
                tbody.appendChild(tr);
            });

            data.outputs.forEach(f => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${f} (Analyzed)</td>
                    <td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>
                `;
                tbody.appendChild(tr);
            });
          }

          async function analyzeVideo(filename) {
            const formData = new FormData();
            formData.append('filename', filename);
            const res = await fetch('/api/video/analyze', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.status === 'started') {
                startStatusPolling();
            } else {
                alert(data.message || 'Error starting analysis');
            }
          }

          let pollInterval;
          function startStatusPolling() {
            document.getElementById('processingArea').style.display = 'block';
            if (pollInterval) clearInterval(pollInterval);

            pollInterval = setInterval(async () => {
                const res = await fetch('/api/video/status');
                const data = await res.json();

                document.getElementById('currentFileName').innerText = data.current_video;
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressText').innerText = data.progress + '%';
                document.getElementById('errorText').innerText = data.error || '';

                if (!data.is_processing && data.progress === 100) {
                    clearInterval(pollInterval);
                    alert('Analysis completed!');
                    refreshFileList();
                } else if (!data.is_processing && data.error) {
                    clearInterval(pollInterval);
                }
            }, 1000);
          }

          fetch('/api/video/status').then(res => res.json()).then(data => {
            if (data.is_processing) startStatusPolling();
          });
        </script>
      </body>
    </html>
    """, media_type="text/html")

def run_fastapi(host, port):
    print("\n" + "="*50, flush=True)
    print("Registered Routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"Path: {route.path:35} | Methods: {route.methods}", flush=True)
    print("="*50 + "\n", flush=True)
    sys.stdout.flush()

    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)

def _primary_output_tensor(hailo_output):
    if isinstance(hailo_output, dict):
        return next(iter(hailo_output.values()))
    if isinstance(hailo_output, (list, tuple)):
        return hailo_output[0]
    return hailo_output

def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """
    Decode the Hailo Segmentation output layer.
    Supports two common Model Zoo layouts:
    - single-channel class-index masks, e.g. (512, 1024, 1) or (1, 512, 1024)
    - multi-channel logits, e.g. (512, 1024, C) or (C, 512, 1024)

    Returns a 2D integer mask of shape (input_h, input_w) containing class indices per pixel.
    """
    global _SEG_OUTPUT_LOGGED

    if hailo_output is None:
        return None

    output_tensor = np.asarray(_primary_output_tensor(hailo_output))
    raw_shape = output_tensor.shape

    if output_tensor.ndim == 4:
        output_tensor = np.squeeze(output_tensor, axis=0) if output_tensor.shape[0] == 1 else output_tensor[0]

    output_tensor = np.squeeze(output_tensor)

    if output_tensor.ndim == 2:
        class_mask = output_tensor
    elif output_tensor.ndim == 3:
        if output_tensor.shape[-1] == 1:
            class_mask = output_tensor[:, :, 0]
        elif output_tensor.shape[0] == 1:
            class_mask = output_tensor[0, :, :]
        elif output_tensor.shape[-1] in (len(CLASSES), 19, 34):
            class_mask = np.argmax(output_tensor, axis=-1)
        elif output_tensor.shape[0] in (len(CLASSES), 19, 34):
            class_mask = np.argmax(output_tensor, axis=0)
        else:
            raise ValueError(f"Unsupported segmentation output shape: {raw_shape}")
    else:
        raise ValueError(f"Unsupported segmentation output shape: {raw_shape}")

    if class_mask.shape == (input_w, input_h) and input_h != input_w:
        class_mask = class_mask.T
    elif class_mask.shape != (input_h, input_w):
        class_mask = cv2.resize(class_mask.astype(np.float32), (input_w, input_h),
                                interpolation=cv2.INTER_NEAREST)

    class_mask = np.nan_to_num(class_mask, nan=0.0, posinf=0.0, neginf=0.0)
    class_mask = np.rint(class_mask).clip(0, 255).astype(np.uint8)

    if not _SEG_OUTPUT_LOGGED:
        unique = np.unique(class_mask)
        preview = unique[:20].tolist()
        print(
            f"[SegFormer] raw output shape={raw_shape}, decoded mask shape={class_mask.shape}, "
            f"labels={preview}{'...' if len(unique) > 20 else ''}",
            flush=True,
        )
        _SEG_OUTPUT_LOGGED = True

    return class_mask


# def draw(image, boxes, scores, classes):
#     for box, score, cl in zip(boxes, scores, classes):
#         top, left, right, bottom = [int(_b) for _b in box]
#         cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
#         cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
#                         (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
def draw_segmentation_mask(image, mask, lb_info):
    """
    Un-letterbox the network's 2D class-index mask, scale it to the original
    frame size with nearest-neighbor (class indices must stay integral), and
    alpha-blend a colored overlay onto `image` in place.

    `lb_info` is the (ratio, dw, dh) tuple returned by preprocess_frame for
    this exact frame; we cannot rely on co_helper.letter_box_info_list[-1]
    because letterbox calls from other threads (e.g. VideoAnalyzer running
    alongside the live preview) interleave into the same list.
    """
    if mask is None:
        return image

    orig_h, orig_w = image.shape[:2]
    H, W = mask.shape

    _, dw, dh = lb_info
    # Replicate letter_box's int rounding so the crop matches the padding it
    # added exactly. dw/dh come back as floats representing HALF the total
    # padding; letter_box used round(dh-0.1) for top and round(dh+0.1) for
    # bottom (and likewise for left/right) to absorb the odd-pixel case.
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    h_end = H - bottom if bottom > 0 else H
    w_end = W - right if right > 0 else W
    content_mask = mask[top:h_end, left:w_end]

    full_scale_mask = cv2.resize(content_mask, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)

    max_label_present = max(len(CLASSES), int(full_scale_mask.max()) + 1)
    colors = np.zeros((max_label_present, 3), dtype=np.uint8)
    palette_len = min(len(CITYSCAPES_PALETTE), len(colors))
    colors[:palette_len] = CITYSCAPES_PALETTE[:palette_len]
    if len(colors) > palette_len:
        rng = np.random.default_rng(42)
        colors[palette_len:] = rng.integers(0, 255, size=(len(colors) - palette_len, 3), dtype=np.uint8)

    color_mask = colors[full_scale_mask]

    alpha = 0.4
    mask_indices = full_scale_mask != 255
    image[mask_indices] = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)[mask_indices]

    return image

def preprocess_frame(frame, co_helper):
    """Letterbox + BGR to RGB. Returns (img, lb_info) where lb_info captures the
    exact ratio + padding used for this frame so the mask can be un-letterboxed
    independent of any shared co_helper state (the helper appends to its own
    list across threads, so relying on its last entry is racy)."""
    if getattr(co_helper, "letter_box_info_list", None) is not None:
        co_helper.letter_box_info_list.clear()
    img, ratio, (dw, dh) = co_helper.letter_box(
        im=frame.copy(),
        new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
        pad_color=(0, 0, 0),
        info_need=True,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, (ratio, dw, dh)

def inference_loop(cap, model, co_helper, is_video_file, target_fps):
    """Capture + preprocess + Hailo inference + draw on full-res frame.
    Pushes annotated frames into frame_buffer; the encode thread takes over from there.

    `target_fps` caps the loop rate so the preview path doesn't starve other
    Hailo/CPU consumers (notably VideoAnalyzer). Hailo can run 100+ fps but
    no human needs a 100 fps preview, and the leftover budget keeps offline
    video analysis healthy. Pass 0 to disable the cap.

    When VideoAnalyzer is running, the loop drops to 1 fps automatically; a frozen preview is fine for a few seconds, and the freed CPU/Hailo
    cycles roughly halve the analyze wall time."""
    fps_counter = 0
    target_period = 1.0 / target_fps if target_fps > 0 else 0
    next_time = time.time()
    try:
        while not stop_event.is_set():
            # Yield to VideoAnalyzer when it's running.
            if video_analyzer.is_processing:
                time.sleep(1.0)
                next_time = time.time()
                continue

            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            processed_img, lb_info = preprocess_frame(frame, co_helper)
            start_time = time.time()
            outputs = model.run(processed_img)
            inference_time = time.time() - start_time

            if outputs is not None:
                # obj, nms = det_config.get()
                mask = post_process_hailo(outputs, 0, 0, IMG_SIZE[1], IMG_SIZE[0])
                if mask is not None:
                    draw_segmentation_mask(frame, mask, lb_info)

            inf_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'Hailo FPS: {fps_counter:.1f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_buffer.push_annotated(frame)

            if target_period > 0:
                now = time.time()
                next_time += target_period
                sleep_for = next_time - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif sleep_for < -target_period:
                    next_time = now + target_period
    finally:
        stop_event.set()


def encode_loop(preview_w, preview_h, jpeg_quality):
    """Take the latest annotated frame, resize for preview, JPEG-encode, publish.
    Runs at its own pace so a slow encode never back-pressures inference."""
    last_v = -1
    while not stop_event.is_set():
        frame, last_v = frame_buffer.wait_annotated(last_v, timeout=1.0)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if preview_w > 0 and preview_h > 0 and (w, h) != (preview_w, preview_h):
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        else:
            preview = frame
        ok, buf = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            frame_buffer.push_jpeg(buf.tobytes())


def main():
    parser = argparse.ArgumentParser(description='SegFormer B0 BN semantic segmentation on RPi5 + Hailo-8 (Web Preview Mode)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .hef model (Hailo Executable Format)')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID (default: 0 for /dev/video0). Use -1 to disable camera and run web-only mode.')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id)')
    parser.add_argument('--class_path', type=str, help='Path to class_config.txt file for dynamic category loading')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    parser.add_argument('--preview_width', type=int, default=1280, help='MJPEG preview width (0 to disable resize). Default 1280.')
    parser.add_argument('--preview_height', type=int, default=720, help='MJPEG preview height (0 to disable resize). Default 720.')
    parser.add_argument('--jpeg_quality', type=int, default=80, help='MJPEG preview JPEG quality 1-100. Default 80.')
    parser.add_argument('--cam_width', type=int, default=1280, help='Requested USB camera width. Default 1280.')
    parser.add_argument('--cam_height', type=int, default=720, help='Requested USB camera height. Default 720.')
    parser.add_argument('--target_fps', type=float, default=30.0, help='Cap live preview inference rate (fps). 0 = uncapped. Default 30.')
    args = parser.parse_args()

    if not HAILO_AVAILABLE:
        print("Error: HailoRT is not available. Install the hailort wheel matching your driver version.")
        return

    if args.class_path:
        load_classes(args.class_path)

    global _global_model, _global_co_helper, IMG_SIZE
    model = HailoInfer(args.model_path)
    # Pull the real (H, W) out of the .hef so any segmentation model (513, 640,
    # 1024, non-square, etc.) works without editing constants. Stored as (W, H)
    # to match the rest of the file.
    IMG_SIZE = (model.input_w, model.input_h)
    print(f"Model input size: {model.input_w}x{model.input_h}", flush=True)
    co_helper = COCO_test_helper(enable_letter_box=True)

    _global_model = model
    _global_co_helper = co_helper
    video_analyzer.set_engine(model, co_helper)

    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web Preview started at http://{args.host}:{args.port}", flush=True)
    print(f"Preview: {args.preview_width}x{args.preview_height} JPEG q={args.jpeg_quality} | target_fps={args.target_fps}", flush=True)
    sys.stdout.flush()

    if args.camera_id == -1 and not args.video_path:
        print("Running in Video Analysis Mode. Access Web UI to process local videos.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            model.release()
        return

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
        capture_source = cap
        is_video_file = True
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Ask the camera to deliver MJPG so USB bandwidth isn't wasted on raw
        # YUYV. Most modern USB webcams do MJPG natively; if not, V4L2 quietly
        # falls back to YUYV and we just lose this win 鈥?no error.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        capture_source = None
        is_video_file = False

    if not cap.isOpened():
        print(f"Error: Cannot open video source (ID: {args.camera_id if not args.video_path else args.video_path})")
        return

    if not is_video_file:
        capture_source = LatestFrameReader(cap).start()

    inf_thread = threading.Thread(target=inference_loop,
                                  args=(capture_source, model, co_helper, is_video_file, args.target_fps),
                                  daemon=True)
    enc_thread = threading.Thread(target=encode_loop,
                                  args=(args.preview_width, args.preview_height, args.jpeg_quality),
                                  daemon=True)
    inf_thread.start()
    enc_thread.start()

    try:
        while inf_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_event.set()
        if not is_video_file:
            capture_source.stop()
        inf_thread.join(timeout=2)
        enc_thread.join(timeout=2)
        cap.release()
        model.release()

if __name__ == '__main__':
    main()

