# Hailo-8 YOLOv8 Docker 联机测试报告

**测试日期**：2026-05-20
**测试模块**：[src/rpi5_hailo8_yolov8/](./)
**测试结论**：通过 ✓

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 主机 | Raspberry Pi 5（reComputer R2x，IP `192.168.10.211`） |
| OS | Debian Bookworm，内核 `6.12.87+rpt-rpi-2712 aarch64` |
| Hailo 设备 | Hailo-8 M.2（PCIe），设备节点 `/dev/hailo0` |
| Hailo 固件 | `4.20.0 (release, app, extended context switch buffer)` |
| 序列号 | `Seeed252600457` |
| 宿主机 native 库 | `/usr/lib/libhailort.so.4.20.0`（来自 `hailo-all` apt 包） |
| 容器内 wheel | `hailort-4.20.0-cp311-cp311-linux_aarch64.whl` |
| 模型 | `model/yolov8n.hef`（Hailo Model Zoo，内置 NMS layer） |
| Docker 镜像 | `rpi5-hailo8-yolov8:latest`（基于 `python:3.11-slim` arm64，1.12 GB） |
| 测试视频 | `video/test.mp4`，3840×2160（4K） |

---

## 2. 关键问题与修复

### 2.1 编译依赖缺失

**现象**：构建镜像时 `pip install hailort-*.whl` 失败：
```
error: command 'gcc' failed: No such file or directory
ERROR: Failed building wheel for netifaces
```

**根因**：`hailort` wheel 间接依赖 `netifaces`，该包在 cp311/aarch64 上无预编译 wheel，需要现场编译。`python:3.11-slim` 没装 gcc 和 Python 头文件。

**修复**：[docker/hailo8/yolov8.dockerfile](../../docker/hailo8/yolov8.dockerfile) 在装 wheel 那一步 apt 装 `build-essential`、`python3-dev`，装完后 `apt-get purge --auto-remove` 清掉，避免镜像膨胀。

### 2.2 容器内找不到 libhailort.so

**现象**：容器启动后立即报：
```
libhailort.so.4.20.0: cannot open shared object file: No such file or directory
```

**根因**：`hailort` wheel 只装了 Python bindings（`hailo_platform/` 模块），native 库 `libhailort.so` 是宿主机 `hailo-all` apt 包提供的（在 `/usr/lib/`），容器里没有。

**修复**：`docker run` 加两条 bind-mount：
```
-v /usr/lib/libhailort.so.4.20.0:/usr/lib/libhailort.so.4.20.0:ro
-v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro
```
README 的所有 `docker run` 示例已同步更新。

### 2.3 后处理 list/ndarray 类型不一致

**现象**：推理第一帧即崩：
```
File "/app/web_detection.py", line 634, in post_process_hailo
    if output.dtype == object:
       ^^^^^^^^^^^^
AttributeError: 'list' object has no attribute 'dtype'
```

**根因**：HailoRT 4.20 在 NMS-by-score 输出格式下返回的是 Python `list`（不是 `np.ndarray`），原代码假设了 ndarray 才有的 `.dtype` 属性。

**修复**：[web_detection.py:610](web_detection.py#L610) `post_process_hailo()` 去掉 `output.dtype` 分支判断 — 三种可能输出（list / object-ndarray / 稠密 float ndarray）的 `output[0]` 都能直接 `enumerate` 得到 per-class 检测，统一处理即可。

---

## 3. 测试用例与结果

### 3.1 容器启动

```bash
sudo docker run -d --name hailo-prod --privileged --net=host \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.20.0:/usr/lib/libhailort.so.4.20.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov8:latest
```

**结果**：通过 ✓
- 容器持续运行无 crash
- 日志输出 `Web Preview started at http://0.0.0.0:8000`
- 14 个 FastAPI 路由全部注册成功
- HailoRT 成功 dlopen `libhailort.so.4.20.0`

### 3.2 根页面 `GET /`

```bash
curl -s -o /dev/null -w "%{http_code} %{size_download}\n" http://localhost:8000/
```

**结果**：通过 ✓ — `HTTP 200, 10710 bytes`（HTML 页面）

### 3.3 配置接口 `GET/POST /api/config`

```bash
curl -s http://localhost:8000/api/config
# {"obj_thresh":0.25,"nms_thresh":0.45}

curl -s -X POST http://localhost:8000/api/config \
     -H "Content-Type: application/json" \
     -d '{"obj_thresh":0.5}'
# {"status":"success"}

curl -s http://localhost:8000/api/config
# {"obj_thresh":0.5,"nms_thresh":0.45}
```

**结果**：通过 ✓ — 读、写、回读均正确

### 3.4 单帧推理 `POST /api/models/yolov8/predict`

输入：从 `video/test.mp4` 提取的 4K 帧（3840×2160）

```bash
curl -X POST http://localhost:8000/api/models/yolov8/predict -F "file=@frame.jpg"
```

**结果**：通过 ✓
- `success: true`
- 检测到 **22** 个目标（21 辆 car + 1 辆 bus）
- 最高置信度 `car @ 0.787`，box `[2108, 1483, 2291, 1651]`
- 坐标已正确反算到原图 3840×2160 像素空间（未停留在 letterboxed 640×640）

### 3.5 MJPEG 视频流 `GET /api/video_feed`

```bash
timeout 3 curl -s http://localhost:8000/api/video_feed -o stream.bin
ls -la stream.bin
```

**结果**：通过 ✓
- 3 秒内拉取 **195 MB**（持续帧）
- 解出单 JPEG 帧 2.37 MB（4K，标注后），SOI/EOI 完整
- 视觉验证：[test_snapshot.jpg](../../test_snapshot.jpg) 北京街景，蓝框 + 红字类名 + 置信度 + 左上角 `Hailo FPS` 计数器全部正确渲染

### 3.6 视频上传 + 异步分析

```bash
curl -X POST http://localhost:8000/api/video/upload -F "file=@test.mp4"
# {"filename":"short.mp4","status":"uploaded"}

curl -X POST http://localhost:8000/api/video/analyze -F filename=short.mp4
# {"status":"started","output":"short_3840x2160_results.mp4"}

# 轮询
curl http://localhost:8000/api/video/status
# {"is_processing":true,"progress":4,...}    # 4 秒后
# {"is_processing":false,"progress":100,...} # 24 秒后

curl -s -o output.mp4 http://localhost:8000/api/video/download/short_3840x2160_results.mp4
# 146 MB 带标注的输出视频
```

**注意**：`/api/video/analyze` 接受 form-data `filename=xxx`，不是 JSON body（与原 RK 版本一致）。

**结果**：通过 ✓
- 上传成功
- 异步任务从 0% 推进到 100%
- 输出文件 146 MB 可下载

### 3.7 视频列表 `GET /api/video/list`

```bash
curl http://localhost:8000/api/video/list
# {"uploads":["short.mp4"],"outputs":["short_3840x2160_results.mp4"]}
```

**结果**：通过 ✓

### 3.8 日志扫描

```bash
sudo docker logs hailo-prod 2>&1 | grep -iE "error|warning|exception|traceback"
```

**结果**：通过 ✓ — 0 个 error/warning/exception 输出（连续运行 18 分钟）

---

## 4. 性能观察

- 4K MJPEG 流 195 MB / 3 s ≈ **65 MB/s**，相当于持续 ~30 fps 的 JPEG-encoded 4K 帧
- 持久化 `InferVStreams` 生效（每帧不重建 pipeline），未观察到 FPS 抖动
- 视频离线分析速度约 ~10 fps（4K 重 IO 占大头，纯推理远快于此）

---

## 5. 复现步骤（验收用）

```bash
# 1. SSH 到 Pi
ssh recomputer@192.168.10.211   # pwd: <your-pi-password>

# 2. 确认环境
ls /dev/hailo0
hailortcli fw-control identify | grep "Firmware Version"   # 4.20.0
sudo find /usr -name "libhailort.so*"                      # 记下路径
ls ~/wrm/reComputer-RK-CV-main/src/rpi5_hailo8_yolov8/model/yolov8n.hef
ls ~/wrm/reComputer-RK-CV-main/src/rpi5_hailo8_yolov8/hailort-packages/hailort-*.whl

# 3. 构建（已在本次测试中完成，镜像 ID 7894939e2e34）
cd ~/wrm/reComputer-RK-CV-main
sudo docker build -f docker/hailo8/yolov8.dockerfile -t rpi5-hailo8-yolov8:latest src/rpi5_hailo8_yolov8/

# 4. 运行
sudo docker run --rm --privileged --net=host \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.20.0:/usr/lib/libhailort.so.4.20.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov8:latest

# 5. 浏览器打开 http://192.168.10.211:8000 即可看到检测画面
```

---

## 6. 已知限制

1. **Hailo 库版本绑定**：bind-mount 路径 `/usr/lib/libhailort.so.4.20.0` 硬编码了固件版本号。如果以后升级宿主机 `hailo-all`，要同步：
   - 替换 `hailort-packages/` 里的 wheel
   - 重建镜像
   - `docker run` 里改 `-v` 路径
2. **`--privileged` 偏宽**：当前为简便起见用了 `--privileged`。生产环境可以收紧到 `--device /dev/hailo0` + 适当的 capabilities（`SYS_RAWIO`），但需要测试 PCIe 驱动是否能跑通。
3. **首帧 `dtype==object` 分支已未走**：HailoRT 4.20 实测返回 Python list；如未来某版本切回 object ndarray，[web_detection.py](web_detection.py) 的 `post_process_hailo` 仍能正常工作（已统一处理）。

---

## 7. 输出物清单

| 路径 | 说明 |
|---|---|
| [docker/hailo8/yolov8.dockerfile](../../docker/hailo8/yolov8.dockerfile) | 已加 `build-essential` 编译 + 清理 |
| [src/rpi5_hailo8_yolov8/web_detection.py](web_detection.py) | `post_process_hailo` 去掉 `dtype` 分支 |
| [src/rpi5_hailo8_yolov8/README.md](README.md) | 加入 `libhailort.so` bind-mount 说明 |
| [src/rpi5_hailo8_yolov8/README_zh.md](README_zh.md) | 同上中文版 |
| [test_snapshot.jpg](../../test_snapshot.jpg) | 测试期间从 MJPEG 流截取的 4K 检测标注帧 |

---

## 8. 性能优化（V2，2026-05-21）

**问题**：V1 是 RK3588 风格的单线程串行流水线，在 Pi 5 上端到端延迟很高。
实测瓶颈分布：

| 阶段 | 4K @ q=95 | 备注 |
|---|---|---|
| Hailo 推理 | 8.5 ms | 画面"Hailo FPS: 116"只反映这一段 |
| 视频解码（4K H.264，cv2 单线程 libav） | 17.4 ms | |
| Letterbox + RGB 转换 | 8.6 ms | |
| 后处理 + 画框 | 0.7 ms | |
| **JPEG 编码（4K q=95）** | **54.7 ms** | **真正的端侧瓶颈** |

网络层面更夸张：本机 localhost 拉 MJPEG 65 MB/s，跨 LAN 到笔记本只有
**0.49 MB/s**（≈ 4 Mbps，WSL/WiFi 链路），原始 4K JPEG 每帧 2.3 MB →
浏览器看到的延迟 5+ 秒。

### 改造内容

| 改造 | 文件 | 收益 |
|---|---|---|
| FrameBuffer 加版本号 + Condition | [web_detection.py:326-374](web_detection.py#L326) | 消除"旧帧堆积"，浏览器永远看到最新帧 |
| capture/inference 与 encode 拆成独立线程 | [web_detection.py:609-686](web_detection.py#L609) | 编码不再背压推理 |
| 推流前缩放到 720p + 默认 q=80 | inf_loop/enc_loop 配合 | 帧体积 2.3 MB → 0.22 MB（**-90%**）|
| USB 摄像头强制 MJPG fourcc | [web_detection.py:730](web_detection.py#L730) | 节省 USB 带宽，避免 YUYV 浪费 |
| `--target_fps` 默认 30 | CLI 新增 | 不让 inf_loop 跑满 CPU，给其它线程留 GIL 时间 |
| `analyze` 期间 inf_loop 自动降到 1 fps | `inference_loop` 内检查 `video_analyzer.is_processing` | 释放 Hailo/CPU 给离线分析 |
| VideoAnalyzer 用 ffmpeg libx264 ultrafast 多线程 | [web_detection.py:107-138](web_detection.py#L107) + Dockerfile 装 `ffmpeg` | cv2 mp4v 4K @ 150ms/帧 → ffmpeg ~100ms/帧 |

> 注：本想用 `avc1`/`h264_v4l2m2m` 走 Pi 硬件 H.264 编码，但 **Pi 5 砍掉了
> 这块硬件**（仅 Pi 4 及更老的设备有），cv2 的 fourcc 自动选码器路径在
> Pi 5 上必失败。必须 subprocess 调外置 ffmpeg + libx264 软件编码。

### 实测对比（4K 测试视频 394 帧）

| 场景 | V1（改造前） | V2（改造后） | 提速 |
|---|---|---|---|
| 实时流跨 LAN 帧率 | **0.2 fps**（4K JPEG 卡死在网络） | **18 fps**（720p JPEG） | **~90×** |
| 实时流跨 LAN 带宽 | 0.49 MB/s | 1.1 MB/s | 2.3× |
| 离线分析（13 秒 4K 视频）| ~120 秒 | **40 秒** | **3×** |
| Pi 本机 MJPEG 帧率 | 28 fps（含重复帧） | 17 fps（去重，每帧都是新的） | 体感"流畅度"显著提升 |

### 未改的部分（及原因）

- **cv2.VideoCapture 单线程 H.264 解码**：libav 多线程解码需要换底层
  API（PyAV 或 ffmpeg subprocess decode），改动太大，留待后续。在 Pi 5 上
  这就限制了离线分析的上限到 ~30 fps（解码 17 ms × 30 = 510 ms/s = 1 核满）。
- **GStreamer/picamera2 路径**：CSI 摄像头才有明显收益，当前测试用 USB
  和文件源，不在范围内。模板留好接口（USB MJPG fourcc 已设），后续接 CSI
  时再切到 `picamera2`。
- **`time.sleep(0.01)` 历史包袱**：V2 全部移除，改为 `Condition.wait_for`，
  CPU 空转和"30 fps 节流的假设"一起去掉了。
