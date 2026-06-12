# FastDepth Depth Estimation on Raspberry Pi 5 + Hailo-8

本模块把 `fast_depth.hef` 单目深度估计模型封装成可在 Raspberry Pi 5 /
reComputer R20 + Hailo-8 上一键部署的 Docker 展示框架。

## 功能

- Web 页面实时预览深度估计结果。
- 将模型输出解码为相对深度图，并以伪彩色深度图覆盖到原画面上。
- REST API 路径为 `/api/models/fast_depth/predict`，返回深度统计值。
- 接口返回保留类似 YOLO 的 `predictions` 列表，同时额外返回 `depth` 统计对象。
- 支持内置视频、USB 摄像头、上传图片/视频帧、离线视频分析。
- 摄像头模式使用最新帧读取，减少 OpenCV/V4L2 缓冲导致的延迟。

## 文件说明

| 路径 | 作用 |
|---|---|
| `web_detection.py` | FastAPI 服务、MJPEG 预览、FastDepth 后处理、离线视频分析 |
| `model/fast_depth.hef` | Hailo-8 深度估计模型 |
| `hailort-packages/*.whl` | 与宿主机驱动 major.minor 匹配的 HailoRT Python wheel |
| `../../docker/hailo8/fast_depth.dockerfile` | 该模块对应的 Dockerfile |
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
cd src/rpi5_hailo8_fast_depth

sudo docker build -f ../../docker/hailo8/fast_depth.dockerfile \
    -t r20-hailo8-fast_depth:latest .
```

## 使用内置视频运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-fast_depth:latest
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
    r20-hailo8-fast_depth:latest \
    python web_detection.py --model_path model/fast_depth.hef --camera_id 0
```

## API

调用示例：

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/fast_depth/predict -F "file=@test.jpg"
```

响应中会包含 `predictions` 和 `depth`，格式示例：

```json
{
  "success": true,
  "predictions": [
    {
      "class": "depth",
      "confidence": 1.0,
      "min_depth": 0.12,
      "max_depth": 8.4,
      "mean_depth": 1.9,
      "median_depth": 1.5,
      "valid_pixels": 921600
    }
  ],
  "depth": {
    "min_depth": 0.12,
    "max_depth": 8.4,
    "mean_depth": 1.9,
    "median_depth": 1.5,
    "valid_pixels": 921600
  }
}
```

注意：这里的深度值是模型输出得到的相对深度，不是米制绝对距离。

## 后处理说明

`post_process_hailo()` 会把 HEF 输出当成连续相对深度图处理。如果输出范围看起来像 logits，会先做 `sigmoid`；否则直接使用模型输出。展示时会按当前帧归一化并渲染为伪彩色深度图。

第一次推理时会打印真实输出 shape 和深度统计：

```text
[FastDepth] raw output shape=..., decoded depth shape=..., min=..., max=...
```

如果真机显示异常，优先把这行日志发出来，用于确认 HailoRT 输出布局。
