# Hailo Model Zoo 模型迁移与一键镜像生成

这份说明用于把 Hailo Model Zoo 里的 `.hef` 模型搬到本项目，并生成可在
Raspberry Pi 5 + Hailo-8 上一键部署的 Docker 镜像。

## 1. 先判断模型类型

目前仓库里已经有两类可复用模板：

| 模型类型 | 推荐模板 | 适合的 Hailo Zoo 输出 |
|---|---|---|
| 目标检测 | `src/rpi5_hailo8_yolov8/` | 内置 NMS，输出接近 `(1, num_classes, max_dets, 5)` |
| 语义分割 | `src/rpi5_hailo8_segformer_b0_bn/` | 输出为逐像素类别分数或 mask，例如 `(1, H, W, num_classes)` 或 `(1, num_classes, H, W)` |

如果是 pose、obb、实例分割、深度估计、分类等模型，可以先用最接近的模板生成骨架，
但必须按 Hailo Zoo 对应 README 的输出张量说明改 `web_detection.py` 里的
`post_process_hailo()` 和 API 返回格式。

## 2. 从 Hailo Model Zoo 准备 `.hef`

在 Hailo Model Zoo 里找到目标模型，下载或编译面向 `hailo8` 的 `.hef`。
文件名建议保持和模型 slug 一致，例如：

```text
yolov8s.hef
segformer_b0_bn.hef
fast_depth.hef
```

注意事项：

- `.hef` 必须是 `hailo8` 目标，不是 `hailo8l` 或其它芯片目标。
- 宿主机 `hailo-all`、容器里的 `hailort` wheel、固件版本的 major.minor 要一致。
- 检测模型尽量使用带 postprocess/NMS 的 Model Zoo 默认 `.hef`，这样可以复用现有检测模板。
- 类别顺序必须和模型训练数据一致；非 COCO/VOC 模型建议额外准备 `class_config.txt`。

## 3. 用脚手架生成项目模块

脚手架会复制模板目录，跳过模板里已有的 `.hef`，生成对应 Dockerfile，并把新的
`.hef` 放到 `model/` 下。

### 目标检测模型

```powershell
python tools/scaffold_hailo_model.py yolov8s `
    --task detection `
    --hef-source D:\Downloads\yolov8s.hef
```

生成：

```text
src/rpi5_hailo8_yolov8s/
docker/hailo8/yolov8s.dockerfile
```

### 语义分割模型

```powershell
python tools/scaffold_hailo_model.py segformer_b0_bn `
    --task segmentation `
    --hef-source D:\Downloads\segformer_b0_bn.hef
```

### 只生成骨架，稍后手动放 `.hef`

```powershell
python tools/scaffold_hailo_model.py yolov8m --task detection
```

然后把模型文件放到：

```text
src/rpi5_hailo8_yolov8m/model/yolov8m.hef
```

## 4. 检查生成后的关键文件

每搬一个模型，重点检查这些位置：

| 文件 | 要检查什么 |
|---|---|
| `docker/hailo8/<model>.dockerfile` | `CMD` 里的 `--model_path model/<model>.hef` 是否正确 |
| `src/rpi5_hailo8_<model>/model/` | 是否存在对应 `.hef` |
| `src/rpi5_hailo8_<model>/hailort-packages/` | 是否有匹配宿主机版本的 `hailort-*.whl` |
| `src/rpi5_hailo8_<model>/web_detection.py` | `post_process_hailo()` 是否适配该模型输出 |
| `src/rpi5_hailo8_<model>/web_detection.py` | API 路径是否是 `/api/models/<model>/predict` |

检测模型如果输出布局和 YOLOv8 模板一致，通常不需要改后处理。
分割模型模板会在启动时读取 `.hef` 的真实输入尺寸，所以不同输入分辨率通常不用手改
`IMG_SIZE`。

## 5. 在树莓派上构建镜像

进入生成的新模块目录：

```bash
cd src/rpi5_hailo8_yolov8s

sudo docker build -f ../../docker/hailo8/yolov8s.dockerfile \
    -t r20-hailo8-yolov8s:latest .
```

## 6. 一键运行

使用内置测试视频：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8s:latest
```

USB 摄像头：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-yolov8s:latest \
    python web_detection.py --model_path model/yolov8s.hef --camera_id 0
```

浏览器打开：

```text
http://<Pi5_IP>:8000
```

## 7. 常见问题

| 现象 | 优先检查 |
|---|---|
| 容器启动时报 HailoRT 相关错误 | 宿主机 `hailo-all`、`.whl`、`libhailort.so` 版本是否一致 |
| 找不到模型文件 | Dockerfile `CMD` 和 `model/` 里的 `.hef` 文件名是否一致 |
| 检测框全错或无结果 | `.hef` 是否带 NMS；不带 NMS 需要重写检测后处理 |
| 分割颜色或类别名错位 | `class_config.txt` 是否按训练类别索引顺序排列 |
| 画面位置偏移 | 模型输入尺寸、letterbox 反算、输出 mask resize 是否一致 |

## 8. 发布到 GHCR 的命名建议

本地验证通过后，镜像 tag 建议统一：

```bash
ghcr.io/<owner>/r20-hailo8-<model>:latest
```

例如：

```bash
ghcr.io/wrm119/r20-hailo8-yolov8s:latest
```
