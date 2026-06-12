# 树莓派 5 + Hailo-8 segformer v3 MobileNet v2 语义分割

面向 **Raspberry Pi 5 + Hailo-8**（reComputer R20 系列）的 segformer v3 MobileNet v2
**语义分割**模块。本模块从 YOLOv5 检测模板 fork 而来（REST 接口路径仍保留
`/api/models/yolov5/...` 以向后兼容），重新对接到 513×513×3 的分割模型。

功能：实时 Web 预览（MJPEG，原图叠加按像素分类的彩色掩膜）、与现有
reComputer-CV 接口约定兼容的 REST API、基于 libx264 ultrafast 的离线视频
批量分析。

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
`segformer_b0_bn.hef`，放进 `model/`：

```bash
cd src/rpi5_hailo8_segformer_b0_bn/model
# 示例路径——具体版本去 Model Zoo 找匹配你 HailoRT 版本的那个
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<version>/hailo8/segformer_b0_bn.hef
```

Model Zoo 默认提供的是在 **PASCAL VOC（21 类，含 `background`）** 上训练的版本，
输入张量形状 `513×513×3`，输出形状 `(1, 513, 513, 21)`，正是
[web_detection.py](web_detection.py) 里 `post_process_hailo()` 期望的格式。
如果你用别的数据集（Cityscapes、ADE20K 等）重新编译，必须额外通过
`--class_path` 把类别索引对到正确的名字。

### 2.2 下载匹配版本的 HailoRT Python wheel

去 Hailo Developer Zone（需注册）下载 `hailort-<version>-cp311-cp311-linux_aarch64.whl`，
其 major.minor 版本号必须与宿主机 `hailortcli fw-control identify` 报告的版本一致。

放到 `hailort-packages/`：

```bash
cd src/rpi5_hailo8_segformer_b0_bn/hailort-packages
# 示例
ls hailort-4.23.0-cp311-cp311-linux_aarch64.whl
```

---

## 3. Docker 方式运行（推荐）

```bash
cd src/rpi5_hailo8_segformer_b0_bn
sudo docker build -f ../../docker/hailo8/segformer_b0_bn.dockerfile -t rpi5-hailo8-segformer:latest .

# 重要：必须把宿主机的 libhailort.so.<X.Y.Z> 挂进容器。
# wheel 里只有 Python bindings，native 库要从宿主机 hailo-all 包来，
# 且 major.minor 版本必须和 wheel 一致。
# 用 `sudo find /usr -name "libhailort.so*"` 查到实际路径。
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-segformer:latest
```

容器启动后，浏览器打开 `http://<Pi5_IP>:8000` 即可看到内置 `video/test.mp4`
循环播放的分割结果。

### 摄像头模式

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    rpi5-hailo8-segformer:latest \
    python web_detection.py --model_path model/segformer_b0_bn.hef --camera_id 0
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
    rpi5-hailo8-segformer:latest \
    python web_detection.py --model_path model/segformer_b0_bn.hef --video_path video/test.mp4 --class_path class_config.txt
```

`class_config.txt` 格式：双引号包裹、逗号分隔。**第一项必须是背景类**
（叠加预览时该类区域保持透明）：

```
"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", ...
```

---

## 4. 不用 Docker 直接运行

```bash
cd src/rpi5_hailo8_segformer_b0_bn
pip install -r requirements.txt
pip install hailort-packages/hailort-*.whl

python web_detection.py --model_path model/segformer_b0_bn.hef --camera_id 0
# 或
python web_detection.py --model_path model/segformer_b0_bn.hef --video_path video/test.mp4
```

---

## 5. REST API
curl -X POST http://localhost:8000/api/models/segformer/predict -F "file=@cat.jpg"

主要接口（完整列表见项目根 [README_zh.md](../../README_zh.md)）：

- `POST /api/models/segformer/predict` — 对上传图片、视频帧或当前摄像头帧做一次推理。
  **路径保留以向后兼容**；响应里的 `predictions` 字段现在是
  `{class, confidence, pixels}` 列表，每一项对应分割掩膜里出现的一个非背景类。
  `confidence` 是该类像素占整张掩膜的比例（0–1），`pixels` 是网络分辨率
  （513×513）下的原始像素数。
- `GET  /api/video_feed` — MJPEG 流，已把分割掩膜叠加到原图上。
- `GET / POST /api/config` — 读/写 `obj_thresh` / `nms_thresh`。两个字段在 API
  里保留以向后兼容，分割流程不会使用它们。
- `POST /api/video/upload`、`POST /api/video/analyze`、`GET /api/video/status`、
  `GET /api/video/download/{filename}` — 本地视频批量分析。

---

## 6. 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model_path` | `.hef` 模型路径 | 必填 |
| `--camera_id` | `/dev/videoN` 序号，`-1` = 纯 Web 模式 | `0` |
| `--video_path` | 视频文件路径（覆盖 `--camera_id`） | 无 |
| `--class_path` | 自定义类名文件 `class_config.txt` | 无（默认 PASCAL VOC 21 类） |
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
  资源让给离线分析。分析完成后自动恢复满速。

---

## 7. 迁移到其它分割模型

把这个目录当模板用：

1. 复制 `src/rpi5_hailo8_segformer_b0_bn/` 整个目录并重命名
2. 从 Model Zoo 下对应 `.hef` 放进 `model/`
3. 如果新模型输出仍是 `(1, H, W, num_classes)` 的逐像素 softmax，只需调整
   `web_detection.py` 里的 `IMG_SIZE` 和默认类别列表
4. 如果通道维在前 `(1, num_classes, H, W)`，`post_process_hailo()` 已经会
   自动识别并转置，不需要改
5. 实例分割、全景分割或其它输出 schema 不同的模型，重写
   `post_process_hailo()`，最终返回形如 `(input_h, input_w)` 的二维类别索引
   掩膜即可
6. 在 `docker/hailo8/` 下新增对应的 `<model>.dockerfile`

---

## 8. 常见问题

| 现象 | 可能原因 |
|---|---|
| 容器里 `Failed to open /dev/hailo0` | 缺 `--device /dev/hailo0:/dev/hailo0` |
| `libhailort.so.<X.Y.Z>: cannot open shared object file` | 缺 `-v /usr/lib/libhailort.so.<X.Y.Z>:...:ro` 挂载。wheel 只装了 Python bindings，`.so` 必须从宿主机拿。 |
| `HailoRT firmware version mismatch` | 宿主机驱动和容器里 wheel 的 major.minor 版本不同 |
| 预览上掩膜位置偏移/被裁切 | `.hef` 输入尺寸和 `IMG_SIZE` 不一致，把它改成 `hef.get_input_vstream_infos()[0].shape[:2]`（默认 `(513, 513)`） |
| 颜色/类别名错位 | `.hef` 不是在 VOC 上训练的。用 `--class_path class_config.txt` 按索引顺序列出实际类别，**第一项必须是 background** |
| 只有个位数 FPS | 多半是每帧重建了 `InferVStreams`——保持 `HailoInfer` 实例长生命周期（[py_utils/hailo_executor.py](py_utils/hailo_executor.py) 默认就是这样做的） |

---

## 9. 性能说明

- 推理耗时由 513×513 的前向占主导；分割解码只是沿通道做一次 `np.argmax`，
  和 Hailo 调用本身比可以忽略不计。
- 掩膜会**用最近邻插值缩回到原帧尺寸**再做 alpha 混合，所以 4K 输入仍能
  呈现清晰的类别边界。
- 推理/编码线程切分沿用 YOLOv5 模板：推理线程把标注帧推进条件变量缓存，
  编码线程从中取最新一帧做 JPEG 编码。慢客户端不会反压到推理线程。

本模块暂未发布单独的验证报告；由于线程模型未变，推理/预览/离线分析流水线
的性能特征参考 YOLOv5 模板即可。
