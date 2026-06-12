# SCRFD Face Detection on Raspberry Pi 5 + Hailo-8

本模块把 Hailo Model Zoo 的 SCRFD 人脸检测模型封装成可在 Raspberry Pi 5 /
reComputer R20 + Hailo-8 上一键部署的 Docker 展示框架。

当前已内置三个 640x640x3 输入的 `.hef`：

| 模型 | 路径 | 说明 |
|---|---|---|
| SCRFD 500M | `model/scrfd_500m.hef` | 默认模型，轻量，适合先跑通 |
| SCRFD 2.5G | `model/scrfd_2.5g.hef` | 中等规模，精度/速度折中 |
| SCRFD 10G | `model/scrfd_10g.hef` | 较大模型，精度更高，速度更慢 |

## 功能

- Web 页面实时预览人脸检测结果。
- 检测框显示类别 `face` 和置信度。
- SCRFD 5 点关键点会以小圆点绘制在人脸上。
- REST API 返回检测框、置信度和 `landmarks`。
- 支持内置视频、USB 摄像头、上传图片/视频帧、离线视频分析。

## 文件说明

| 路径 | 作用 |
|---|---|
| `web_detection.py` | FastAPI 服务、MJPEG 预览、SCRFD 后处理、离线视频分析 |
| `model/*.hef` | SCRFD Hailo-8 模型 |
| `hailort-packages/*.whl` | 与宿主机驱动 major.minor 匹配的 HailoRT Python wheel |
| `../../docker/hailo8/scrfd.dockerfile` | 该模块对应的 Dockerfile |
| `video/test.mp4` | 内置测试视频 |

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
cd src/rpi5_hailo8_scrfd

sudo docker build -f ../../docker/hailo8/scrfd.dockerfile \
    -t r20-hailo8-scrfd:latest .
```

## 使用内置视频运行

默认运行 `model/scrfd_500m.hef`：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest
```

浏览器打开：

```text
http://<Pi5_IP>:8000
```

## 切换 SCRFD 模型

运行 2.5G：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest \
    python web_detection.py --model_path model/scrfd_2.5g.hef --video_path video/test.mp4
```

运行 10G：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest \
    python web_detection.py --model_path model/scrfd_10g.hef --video_path video/test.mp4
```

## 使用 USB 摄像头运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scrfd:latest \
    python web_detection.py --model_path model/scrfd_500m.hef --camera_id 0
```

## API

| Endpoint | 方法 | 说明 |
|---|---|---|
| `/api/models/scrfd/predict` | POST | 对上传图片、视频指定帧或摄像头当前帧做一次人脸检测 |
| `/api/video_feed` | GET | MJPEG 实时预览流 |
| `/api/config` | GET / POST | 读取或更新置信度 / SCRFD NMS 阈值 |
| `/api/video/upload` | POST | 上传视频用于离线分析 |
| `/api/video/analyze` | POST | 启动离线分析 |
| `/api/video/status` | GET | 轮询分析进度 |
| `/api/video/list` | GET | 列出上传和输出文件 |
| `/api/video/download/{filename}` | GET | 下载处理结果 |

调用示例：

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/scrfd/predict -F "file=@test.jpg"
```

响应示例：

```json
{
  "success": true,
  "predictions": [
    {
      "class": "face",
      "confidence": 0.93,
      "box": {"x1": 120, "y1": 80, "x2": 260, "y2": 240},
      "landmarks": [
        {"x": 155, "y": 130},
        {"x": 220, "y": 130}
      ]
    }
  ]
}
```

## 后处理说明

SCRFD `.hef` 输出不是 YOLO 的片上 NMS 格式，而是 raw score / bbox /
landmark 分支。本模块已经在 `post_process_hailo()` 中实现：

- stride 8 / 16 / 32 三层输出自动识别；
- distance-to-anchor-center bbox decode；
- 5 点 landmark decode；
- 置信度过滤和 NMS；
- letterbox 坐标反算回原图。

如果你后面换其它人脸模型，重点检查输出分支 shape 和 decode 方式是否仍与 SCRFD 一致。
