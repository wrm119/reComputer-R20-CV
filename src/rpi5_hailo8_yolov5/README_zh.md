# 树莓派 5 + Hailo-8 YOLOv5 模板

这是面向 **Raspberry Pi 5 + Hailo-8**（reComputer R20 系列）的 YOLOv5 目标检测模块。
按**模板**方式组织：复制目录、换 `.hef`，即可迁移到 Hailo Model Zoo 的其它模型
（yolov5s/m、yolov8、yolov11、yolov5-seg 等）。

功能：实时 Web 预览（带检测框的 MJPEG 流）、兼容 Ultralytics Cloud API 的 REST
接口、用 libx264 ultrafast 做离线视频批量分析。

---

## 1. 宿主机准备（Raspberry Pi 5）

容器要访问 Hailo-8 芯片，需要先在宿主机装好 PCIe 驱动、固件和用户态工具。

```bash
# Raspberry Pi OS Bookworm（推荐）
sudo apt update
sudo apt install hailo-all
sudo reboot

# 校验芯片识别
hailortcli fw-control identify
ls /dev/hailo0
```

### 安装 Docker

在开发板上执行以下命令安装 Docker：

```bash
# 下载安装脚本
curl -fsSL https://get.docker.com -o get-docker.sh
# 使用阿里云镜像源安装
sudo sh get-docker.sh --mirror Aliyun
# 启动 Docker，并设置开机自启动
sudo systemctl enable docker
sudo systemctl start docker
```

`hailortcli fw-control identify` 应该打印出板卡信息和固件版本。**记住这个固件版本**
——容器里装的 `hailort` wheel 必须和它对得上。

---

## 2. 资源下载

### 2.1 下载模型

从 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) 下载预编译的
`yolov5s.hef`，放进 `model/`：

```bash
cd src/rpi5_hailo8_yolov5/model
# 示例路径——具体版本去 Model Zoo 找匹配你 HailoRT 版本的那个
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<version>/hailo8/yolov5s.hef
```

Model Zoo 默认的 `yolov5s.hef` **内置了 NMS 后处理层**，这正是
[web_detection.py](web_detection.py) 里 `post_process_hailo()` 期望的格式。
如果你自己编译 `.hef` 时关掉了 NMS layer，输出会变成 3 分支原始张量，需要
重新写 DFL + box 解码。

### 2.2 下载匹配版本的 HailoRT Python wheel

去 Hailo Developer Zone（需注册）下载 `hailort-<version>-cp311-cp311-linux_aarch64.whl`，
其 major.minor 版本号必须与宿主机 `hailortcli fw-control identify` 报告的版本一致。

放到 `hailort-packages/`：

```bash
cd src/rpi5_hailo8_yolov5/hailort-packages
# 示例
ls hailort-4.23.0-cp311-cp311-linux_aarch64.whl
```

---

## 3. Docker 方式运行（推荐）

```bash
cd src/rpi5_hailo8_yolov5
sudo docker build -f ../../docker/hailo8/yolov5.dockerfile -t rpi5-hailo8-yolov5:latest .

# 重要：必须把宿主机的 libhailort.so.<X.Y.Z> 挂进容器。
# wheel 里只有 Python bindings，native 库要从宿主机 hailo-all 包来，
# 且 major.minor 版本必须和 wheel 一致。
# 用 `sudo find /usr -name "libhailort.so*"` 查到实际路径。
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov5:latest
```

容器启动后，浏览器打开 `http://<Pi5_IP>:8000` 即可看到内置 `video/test.mp4`
循环播放的检测结果。

### 摄像头模式

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov5:latest \
    python web_detection.py --model_path model/yolov5s.hef --camera_id 0
```

### 自定义类别

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -v $(pwd)/class_config.txt:/app/class_config.txt \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-yolov5:latest \
    python web_detection.py --model_path model/yolov5s.hef --video_path video/test.mp4 --class_path class_config.txt
```

`class_config.txt` 格式：双引号包裹、逗号分隔的类别名
`"person", "bicycle", "car"`

---

## 4. 不用 Docker 直接运行

```bash
cd src/rpi5_hailo8_yolov5
pip install -r requirements.txt
pip install hailort-packages/hailort-*.whl

python web_detection.py --model_path model/yolov5s.hef --camera_id 0
# 或
python web_detection.py --model_path model/yolov5s.hef --video_path video/test.mp4
```

---

## 5. REST API

主要接口（完整列表见项目根 [README_zh.md](../../README_zh.md)）：

- `POST /api/models/yolov5/predict` — 对上传图片、视频帧或当前摄像头帧做一次推理
- `GET  /api/video_feed` — MJPEG 流，已叠加检测框
- `GET / POST /api/config` — 读/写 `obj_thresh` / `nms_thresh`
- `POST /api/video/upload`、`POST /api/video/analyze`、`GET /api/video/status`、
  `GET /api/video/download/{filename}` — 本地视频批量分析

`nms_thresh` 仍保留以兼容原 API，但 Model Zoo 的 `yolov5s.hef` 已经在芯片上做完
NMS，所以这个滑杆实际上只是个额外的置信度过滤辅助旋钮。`obj_thresh` 会在芯片
返回后做客户端过滤。

---

## 6. 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model_path` | `.hef` 模型路径 | 必填 |
| `--camera_id` | `/dev/videoN` 序号，`-1` = 纯 Web 模式 | `0` |
| `--video_path` | 视频文件路径（覆盖 `--camera_id`） | 无 |
| `--class_path` | 自定义类名文件 `class_config.txt` | 无（默认 COCO 80 类） |
| `--host` | Web 服务监听地址 | `0.0.0.0` |
| `--port` | Web 服务端口 | `8000` |
| `--preview_width` | MJPEG 预览缩放宽度（0 = 原分辨率） | `1280` |
| `--preview_height` | MJPEG 预览缩放高度（0 = 原分辨率） | `720` |
| `--jpeg_quality` | MJPEG 预览 JPEG 质量 1-100 | `80` |
| `--cam_width` | USB 摄像头请求宽度 | `1280` |
| `--cam_height` | USB 摄像头请求高度 | `720` |
| `--target_fps` | 实时预览推理速率上限（0 = 不限） | `30` |

调优建议：
- WiFi 慢？把 `--preview_width/height` 调更小（比如 640×360）或者把
  `--jpeg_quality` 降到 60。每一步大约能砍掉一半字节/帧。
- 千兆有线/本机？设 `--preview_width 0` 关掉缩放，直接推原分辨率。
- MJPEG 接口用条件变量推送最新帧，慢客户端不会造成"旧帧堆积"，浏览器
  看到的永远是接近"现在"的画面。
- `/api/video/analyze` 跑的时候，实时预览会自动降到 1 fps，把 Hailo/CPU
  资源让给离线分析（4K 源大约提速 2 倍）。分析完成后自动恢复满速。

---

## 7. 迁移到其它模型（yolov8、yolov11、yolov5-seg 等）

把这个目录当模板用：

1. 复制 `src/rpi5_hailo8_yolov5/` 整个目录，重命名为如 `rpi5_hailo8_yolov5/`
2. 从 Model Zoo 下对应 `.hef` 放进 `model/`
3. 如果新模型**输出布局相同**（带 NMS、`(1, num_classes, max_dets, 5)`），其它都不用改
4. 对 seg/pose/obb 或不带 NMS 层的模型，重写 `post_process_hailo()` 以适配实际
   输出张量。Model Zoo 中每个模型的 README 都有输出 spec 说明
5. 在 `docker/hailo8/` 下新增对应的 `<model>.dockerfile`

---

## 8. 常见问题

| 现象 | 可能原因 |
|---|---|
| 容器里 `Failed to open /dev/hailo0` | 缺 `--device /dev/hailo0:/dev/hailo0` |
| `libhailort.so.<X.Y.Z>: cannot open shared object file` | 缺 `-v /usr/lib/libhailort.so.<X.Y.Z>:...:ro` 挂载。wheel 只装了 Python bindings，`.so` 必须从宿主机拿。 |
| `HailoRT firmware version mismatch` | 宿主机驱动和容器里 wheel 的 major.minor 版本不同 |
| 检测框位置偏 | `.hef` 输入分辨率不是 640x640，修改 `web_detection.py` 里的 `IMG_SIZE` 与 `hef.get_input_vstream_infos()[0].shape[:2]` 一致 |
| 只有个位数 FPS | 多半是每帧重建了 `InferVStreams`——保持 `HailoInfer` 实例长生命周期（[py_utils/hailo_executor.py](py_utils/hailo_executor.py) 默认就是这样做的） |
| `output.dtype == object` 分支始终走/不走 | HailoRT 版本不同导致；两个分支逻辑等价，不影响检测结果 |

---

## 9. 验证与性能

完整验证日志见 [TEST_REPORT.md](TEST_REPORT.md)：

- 所有接口在 Pi 5 + Hailo-8 上用 4K 源视频走通端到端测试
- Hailo 推理 8.5 ms/帧（yolov5s.hef）
- 跨 LAN MJPEG：V2 拆线程 + 720p 缩放后约 18 fps（比 V1 提速 90 倍）
- 离线分析：394 帧 4K 视频 40 秒（比 V1 提速 3 倍，ffmpeg libx264 ultrafast）
- Pi 5 注意：**没有** `h264_v4l2m2m` 硬件编码器（Pi 4 才有）——只能软件编码
