# Person Attribute ResNet on Raspberry Pi 5 + Hailo-8

本模块把 `person_attr_resnet_v1_18.hef` 人体属性识别模型封装成可在
Raspberry Pi 5 / CM5 + Hailo-8 上一键部署的 Docker 项目。

这个模型不是检测模型，而是 Person Attribute 分类模型。它更适合输入单个人体
裁剪图，或者画面中主体人物比较明显的图像，然后输出 PETA 人体属性，例如年龄段、
性别、帽子、Logo、长发、围巾、塑料袋、墨镜等。

## 模型信息

- 任务：Person Attribute
- 输入尺寸：`224x224x3`
- 输出：35 个 PETA 属性 logits / scores
- 后处理：`sigmoid + threshold`
- 默认阈值：`0.70`，与 Hailo TAPPAS 官方 person attributes 后处理一致
- REST API：`/api/models/person_attr_resnet/predict`

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `web_detection.py` | FastAPI 服务、MJPEG 预览、人体属性后处理 |
| `model/person_attr_resnet_v1_18.hef` | Hailo-8 HEF 模型 |
| `requirements.txt` | Python 依赖 |
| `../../docker/hailo8/person_attr_resnet.dockerfile` | Dockerfile |

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
cd src/rpi5_hailo8_person_attr_resnet

sudo docker build -f ../../docker/hailo8/person_attr_resnet.dockerfile \
    -t r20-hailo8-person_attr_resnet:latest .
```

## 运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-person_attr_resnet:latest
```

浏览器打开：

```text
http://<Pi5_IP>:8000
```

## 摄像头运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-person_attr_resnet:latest \
    python web_detection.py --model_path model/person_attr_resnet_v1_18.hef --camera_id 0
```

## API 调用

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/person_attr_resnet/predict \
    -F "file=@person.jpg"
```

返回值包含：

- `predictions`：过滤后的属性列表
- `attributes`：同样的属性列表，方便前端或脚本读取
- `threshold`：当前阈值
- `image`：带属性文字的 base64 JPEG

## 调试说明

首次推理会打印：

```text
[PersonAttrResNet] output=..., raw shape=..., decoded attrs=..., min=..., max=...
```

如果属性几乎都不显示，可以先把页面阈值从 `0.70` 调低到 `0.50` 左右观察。
如果需要多人场景的人体属性，建议前面先接一个 person detector，再把每个人的
crop 输入到本模型。
