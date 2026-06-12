# YOLOv8 Pose on Raspberry Pi 5 + Hailo-8

本模块把 YOLOv8 Pose 姿态估计模型封装成可在 Raspberry Pi 5 /
reComputer R20 + Hailo-8 上一键部署的 Docker 展示框架。

当前已内置两个 640x640x3 输入的 HEF：

| 模型 | 路径 | 说明 |
|---|---|---|
| YOLOv8s Pose | `model/yolov8s_pose.hef` | 默认模型，速度优先 |
| YOLOv8m Pose | `model/yolov8m_pose.hef` | 更大模型，精度更高，速度更慢 |

## 功能

- Web 页面实时预览人体姿态估计结果。
- 绘制 person 检测框、置信度、COCO 17 个关键点和骨架连线。
- REST API 返回检测框、置信度和 `keypoints`。
- 支持内置视频、USB 摄像头、上传图片/视频帧、离线视频分析。
- 实时摄像头路径使用最新帧读取，减少 OpenCV/V4L2 缓冲导致的延迟。

## 前提条件

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

## 构建镜像

```bash
cd src/rpi5_hailo8_yolov8_pose

sudo docker build -f ../../docker/hailo8/yolov8_pose.dockerfile \
    -t r20-hailo8-yolov8_pose:latest .
```

## 使用内置视频运行

默认运行 `model/yolov8s_pose.hef`：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8_pose:latest
```

浏览器打开：

```text
http://<Pi5_IP>:8000
```

## 使用 USB 摄像头运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8_pose:latest \
    python web_detection.py --model_path model/yolov8s_pose.hef --camera_id 0
```

## 切换 YOLOv8m Pose

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8_pose:latest \
    python web_detection.py --model_path model/yolov8m_pose.hef --video_path video/test.mp4
```

## API

| Endpoint | 方法 | 说明 |
|---|---|---|
| `/api/models/yolov8_pose/predict` | POST | 对上传图片、视频指定帧或摄像头当前帧做一次姿态估计 |
| `/api/video_feed` | GET | MJPEG 实时预览流 |
| `/api/config` | GET / POST | 读取或更新置信度阈值 |
| `/api/video/upload` | POST | 上传视频用于离线分析 |
| `/api/video/analyze` | POST | 启动离线分析 |
| `/api/video/status` | GET | 轮询分析进度 |
| `/api/video/download/{filename}` | GET | 下载处理结果 |

调用示例：

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8_pose/predict -F "file=@test.jpg"
```

响应中的每个 `prediction` 包含：

```json
{
  "class": "person",
  "confidence": 0.91,
  "box": {"x1": 120, "y1": 80, "x2": 360, "y2": 520},
  "keypoints": [
    {"x": 180, "y": 120, "score": 0.87}
  ]
}
```

## 后处理说明

`post_process_hailo()` 期望 HEF 使用 Hailo 内置 NMS，检测行格式为：

```text
[ymin, xmin, ymax, xmax, score, kpt0_x, kpt0_y, kpt0_score, ..., kpt16_x, kpt16_y, kpt16_score]
```

第一次推理时会打印原始输出类型和 shape：

```text
[YOLOv8 Pose] raw output type=..., shape=...
```

如果真机上没有框或关键点错位，优先把这行日志发出来，用来确认 HailoRT 返回的 pose 输出布局。
