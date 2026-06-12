# reComputer-R20-CV

[English](./README.md) | [中文]

面向 **Raspberry Pi 5 + Hailo-8**（reComputer R20 系列）的工业级计算机视觉参考实现。
基于 YOLOv8 的目标检测，配套实时 MJPEG 预览、REST API 与离线视频批量分析 —— 围绕
Pi 5 的真实硬件约束设计（无 `h264_v4l2m2m` 硬件编码器、LAN 带宽有限、PCIe 接的
Hailo-8 加速卡）。

`src/` 下的单一模块按**模板**方式组织：复制目录、换 `.hef`，即可迁移到 Hailo
Model Zoo 的其它模型（yolov8s/m、yolov5、yolov8-seg 等）。

---

## 硬件平台

| 项 | 值 |
|---|---|
| 主板 | Raspberry Pi 5（reComputer R20 系列载板） |
| 加速器 | Hailo-8 M.2（PCIe），设备节点 `/dev/hailo0` |
| 系统 | Raspberry Pi OS Bookworm，内核 6.12+ aarch64 |
| 宿主机驱动 | `hailo-all` apt 包（含 PCIe 驱动、固件、`libhailort.so`） |
| HailoRT | 已验证 4.23.x —— 宿主机驱动 / 固件 / 容器 wheel **major.minor 必须一致** |

---

## 快速开始（一条命令，使用预构建镜像）

发布的镜像已经包含源码、HailoRT wheel、ffmpeg 和三个 Model Zoo 权重
（`yolov8n/s/m`）。宿主机只需要装好 Hailo 工具链即可。

### 1. 宿主机准备（首次，一次性）

#### 安装 Docker

在开发板上运行以下命令安装 Docker：

```bash
# 下载安装脚本
curl -fsSL https://get.docker.com -o get-docker.sh
# 使用阿里云镜像源安装
sudo sh get-docker.sh --mirror Aliyun
# 启动 Docker 并配置开机自启
sudo systemctl enable docker
sudo systemctl start docker
```

#### 安装 Hailo 工具链

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

# 重启后校验芯片识别 + 记下固件版本
hailortcli fw-control identify     # 应该报告 4.23.0
ls /dev/hailo0
```

### 2. 一条命令运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/wrm119/r20-hailo8-yolov8:latest
```

首次运行 Docker 会自动拉取镜像（约 1.8 GB）。之后容器会循环播放内置的
`video/test.mp4` 并在 8000 端口提供 Web 界面，浏览器打开
`http://<Pi5_IP>:8000` 即可看到检测画面。

> **为什么需要 `libhailort.so` bind-mount？** 镜像里只装了 Python 绑定，native
> 库必须从宿主机 `hailo-all` 包来。如果你的固件版本不是 `4.23.0`，把上面两条
> `-v` 里的 `4.23.0` 改成 `hailortcli fw-control identify` 报告的版本（若
> major.minor 不一致，还需要从源码用对应版本 wheel 重新构建镜像）。

### USB 摄像头模式

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/wrm119/r20-hailo8-yolov8:latest \
    python web_detection.py --model_path model/yolov8n.hef --camera_id 0
```

---

## 目录结构

```text
reComputer-R20-CV/
├── docker/hailo8/
│   ├── yolov8.dockerfile           # 镜像: python:3.11-slim arm64 + ffmpeg + HailoRT wheel（默认 yolov8n）
│   ├── yolov5.dockerfile           # 同基础镜像，默认 yolov5s
│   └── yolov11.dockerfile          # 同基础镜像，默认 yolov11n
└── src/
    ├── rpi5_hailo8_yolov8/         # 参考模块——已端到端验证（详见 TEST_REPORT.md）
    ├── rpi5_hailo8_yolov5/         # 兄弟模块，同一套流水线 + yolov5 .hef 权重
    └── rpi5_hailo8_yolov11/        # 兄弟模块，同一套流水线 + yolov11 .hef 权重

# 各模块内部布局一致：
src/rpi5_hailo8_yolovN/
    ├── web_detection.py            # FastAPI + 推理/编码线程化流水线
    ├── py_utils/
    │   ├── hailo_executor.py       # HailoRT 封装，长生命周期 InferVStreams
    │   └── coco_utils.py           # Letterbox + 检测框坐标反算
    ├── model/                      # `.hef` 权重（已打进镜像）
    ├── hailort-packages/           # HailoRT wheel（已打进镜像）
    ├── video/test.mp4              # 内置示例视频
    ├── requirements.txt
    ├── README.md / README_zh.md    # 模块详细文档：部署、命令行参数、故障排查
    └── TEST_REPORT.md              # 验证日志（v8 为完整版，v5/v11 为占位）
```

---

## 从源码构建（用于二次开发）

需要改代码、换其它 `.hef`、或针对不同 HailoRT 版本重新构建时：

```bash
git clone https://github.com/wrm119/reComputer-R20-CV.git
cd reComputer-R20-CV/src/rpi5_hailo8_yolov8

# 需要换资源就替换；镜像里已经有默认的不用动
# ls model/                   # .hef 模型
# ls hailort-packages/        # hailort wheel

sudo docker build -f ../../docker/hailo8/yolov8.dockerfile \
    -t r20-hailo8-yolov8:latest .

# 与上面 quick start 同样的 run 命令，只是把镜像名改成本地 tag
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8:latest
```

---

## REST API

所有接口监听容器内的 `8000` 端口；用 `--net=host` 启动时可直接通过
`http://<Pi5_IP>:8000` 访问。

| Endpoint | 方法 | 说明 |
|---|---|---|
| `/api/models/yolov8/predict` | POST | 对上传图片、视频指定帧或摄像头当前帧做一次推理 |
| `/api/video_feed` | GET | 带检测框的 MJPEG 实时流，可直接嵌入 `<img>` |
| `/api/config` | GET / POST | 读取或更新 `obj_thresh` / `nms_thresh` |
| `/api/video/upload` | POST | 上传视频用于批量分析 |
| `/api/video/analyze` | POST | 启动离线分析任务（form-data `filename=...`） |
| `/api/video/status` | GET | 轮询任务进度 |
| `/api/video/list` | GET | 列出已上传源文件和完成的输出 |
| `/api/video/download/{filename}` | GET | 下载已标注的输出 |

### 推理调用示例

```bash
# 上传图片
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict -F "file=@cat.jpg"

# 上传视频并指定时间点取帧（单位：秒）
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict \
    -F "video=@test.mp4" -F "timestamp=5.5"

# 摄像头当前帧
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict -F "realtime=true"

# 单次请求覆盖阈值
curl -X POST http://<Pi5_IP>:8000/api/models/yolov8/predict \
    -F "file=@cat.jpg" -F "conf=0.5" -F "iou=0.4"
```

响应格式：

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "car",
      "confidence": 0.787,
      "box": { "x1": 2108, "y1": 1483, "x2": 2291, "y2": 1651 }
    }
  ],
  "image": { "width": 3840, "height": 2160 }
}
```

嵌入实时流到任意 HTML 页面：

```html
<img src="http://<Pi5_IP>:8000/api/video_feed">
```

### 动态阈值更新

```bash
# 读取当前值
curl http://<Pi5_IP>:8000/api/config
# {"obj_thresh":0.25,"nms_thresh":0.45}

# 更新（两个字段都可选）
curl -X POST http://<Pi5_IP>:8000/api/config \
     -H "Content-Type: application/json" \
     -d '{"obj_thresh":0.4}'
```

> `nms_thresh` 是为兼容老 API 保留的，Model Zoo 的 `yolov8n.hef` 已经在芯片上做完
> NMS，所以这个值只起一个额外的置信度过滤作用。

---

## 迁移到其它模型

如果要持续从 Hailo Model Zoo 搬模型，推荐先看这份流程文档：
[docs/HAILO_MODEL_PORTING_zh.md](docs/HAILO_MODEL_PORTING_zh.md)。里面包含
`.hef` 准备、模板选择、脚手架生成模块、Docker 构建和运行命令。

把 [src/rpi5_hailo8_yolov8/](src/rpi5_hailo8_yolov8/) 当模板：

1. 复制整个目录，重命名（如 `rpi5_hailo8_yolov8_seg/`）。
2. 把新 `.hef` 放进 `model/`。
3. 如果新模型**输出布局相同**（带 NMS、`(1, num_classes, max_dets, 5)`），其它都
   不用改。
4. 对 seg / pose / obb / 不带 NMS 层的模型，按实际张量 spec 重写
   `web_detection.py` 里的 `post_process_hailo()`。Hailo Model Zoo 每个模型的
   README 都有输出格式说明。
5. 在 `docker/hailo8/` 下加一个对应的 `<model>.dockerfile`。

完整步骤：
[src/rpi5_hailo8_yolov8/README_zh.md 第 7 节](src/rpi5_hailo8_yolov8/README_zh.md#7-迁移到其它模型yolov5yolov8-seg-等)。

---

## 详细文档

各模块部署、命令行参数、故障排查：

- [src/rpi5_hailo8_yolov8/README_zh.md](src/rpi5_hailo8_yolov8/README_zh.md) —— YOLOv8（中文）/ [English](src/rpi5_hailo8_yolov8/README.md)
- [src/rpi5_hailo8_yolov5/README_zh.md](src/rpi5_hailo8_yolov5/README_zh.md) —— YOLOv5（中文）/ [English](src/rpi5_hailo8_yolov5/README.md)
- [src/rpi5_hailo8_yolov11/README_zh.md](src/rpi5_hailo8_yolov11/README_zh.md) —— YOLOv11（中文）/ [English](src/rpi5_hailo8_yolov11/README.md)

验证报告：

- [src/rpi5_hailo8_yolov8/TEST_REPORT.md](src/rpi5_hailo8_yolov8/TEST_REPORT.md) —— 完整端到端验证日志 + V1→V2 性能分析
- v5 / v11 仅有占位 TEST_REPORT，待真机跑通验证清单后回填
