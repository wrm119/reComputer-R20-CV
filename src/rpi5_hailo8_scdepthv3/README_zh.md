# SCDepthV3 Depth Estimation on Raspberry Pi 5 + Hailo-8

本模块把 `scdepthv3.hef` 单目深度估计模型封装成可在 Raspberry Pi 5 /
reComputer R20 + Hailo-8 上一键部署的 Docker 展示框架。

## 功能

- Web 页面实时预览深度估计结果。
- 将模型输出解码为相对深度图，并以伪彩色深度图覆盖到原画面上。
- REST API 路径为 `/api/models/scdepthv3/predict`，返回深度统计值。
- 支持内置视频、USB 摄像头、上传图片/视频帧、离线视频分析。
- 摄像头模式使用最新帧读取，减少 OpenCV/V4L2 缓冲导致的延迟。

## 文件说明

| 路径 | 作用 |
|---|---|
| `web_detection.py` | FastAPI 服务、MJPEG 预览、SCDepthV3 后处理、离线视频分析 |
| `model/scdepthv3.hef` | Hailo-8 深度估计模型 |
| `hailort-packages/*.whl` | 与宿主机驱动 major.minor 匹配的 HailoRT Python wheel |
| `../../docker/hailo8/scdepthv3.dockerfile` | 该模块对应的 Dockerfile |
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
cd src/rpi5_hailo8_scdepthv3

sudo docker build -f ../../docker/hailo8/scdepthv3.dockerfile \
    -t r20-hailo8-scdepthv3:latest .
```

## 使用内置视频运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-scdepthv3:latest
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
    r20-hailo8-scdepthv3:latest \
    python web_detection.py --model_path model/scdepthv3.hef --camera_id 0
```

## API

调用示例：

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/scdepthv3/predict -F "file=@test.jpg"
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

注意：这里的深度值是模型输出转换得到的相对深度，不是米制绝对距离。

## 后处理说明

`post_process_hailo()` 会把 HEF 的 raw 输出解码为二维深度图：

```text
depth = 1 / (sigmoid(raw) * 10 + 0.009)
```

第一次推理时会打印真实输出 shape 和深度统计：

```text
[SCDepthV3] raw output shape=..., decoded depth shape=..., min=..., max=...
```

如果真机显示异常，优先把这行日志发出来，用于确认 HailoRT 输出布局。
