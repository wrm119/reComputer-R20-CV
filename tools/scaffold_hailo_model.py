#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scaffold a deployable RPi5 + Hailo-8 model module from local templates."""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DOCKER_DIR = ROOT / "docker" / "hailo8"

TEMPLATES = {
    "detection": "rpi5_hailo8_yolov8",
    "segmentation": "rpi5_hailo8_segformer_b0_bn",
}

TEMPLATE_DEFAULTS = {
    "rpi5_hailo8_yolov8": {
        "dockerfile": "yolov8.dockerfile",
        "hef": "yolov8n.hef",
        "api": "yolov8",
        "title": "YOLOv8 object detection",
    },
    "rpi5_hailo8_segformer_b0_bn": {
        "dockerfile": "segformer_b0_bn.dockerfile",
        "hef": "segformer_b0_bn.hef",
        "api": "segformer",
        "title": "UNNet MobileNet v2 semantic segmentation",
    },
}

DEFAULT_SKIP_PATTERNS = (
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "workspace",
    "*.hef",
)


def kebab_to_snake(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def slug_to_title(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create src/rpi5_hailo8_<model>/ and docker/hailo8/<model>.dockerfile "
            "from an existing Hailo-8 deployment template."
        )
    )
    parser.add_argument(
        "model",
        help="Model slug from Hailo Model Zoo, for example yolov8s, yolov8m, fast_depth.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TEMPLATES),
        default="detection",
        help="Template family to use. Use detection for NMS object detectors, segmentation for per-pixel masks.",
    )
    parser.add_argument(
        "--template",
        help="Explicit template module under src/. Overrides --task.",
    )
    parser.add_argument(
        "--hef-source",
        type=Path,
        help="Optional path to a downloaded .hef file to copy into the new model/ directory.",
    )
    parser.add_argument(
        "--hef-name",
        help="Name expected inside model/. Defaults to <model>.hef or the basename of --hef-source.",
    )
    parser.add_argument(
        "--api-name",
        help="URL segment for /api/models/<api-name>/predict. Defaults to the model slug.",
    )
    parser.add_argument(
        "--image-prefix",
        default="r20-hailo8",
        help="Docker image prefix used in the printed build command.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing generated module and dockerfile.",
    )
    parser.add_argument(
        "--keep-template-hefs",
        action="store_true",
        help="Copy .hef files from the template. By default they are skipped.",
    )
    parser.add_argument(
        "--keep-template-readmes",
        action="store_true",
        help="Keep template README.md / README_zh.md with text replacements. By default fresh READMEs are generated.",
    )
    return parser.parse_args()


def ignore_factory(keep_template_hefs: bool):
    patterns = list(DEFAULT_SKIP_PATTERNS)
    if keep_template_hefs:
        patterns.remove("*.hef")

    def ignore(_: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                ignored.add(name)
        return ignored

    return ignore


def replace_text(path: Path, replacements: dict[str, str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)

    if updated != text:
        path.write_text(updated, encoding="utf-8", newline="")


def copy_template(src: Path, dst: Path, keep_template_hefs: bool, force: bool) -> None:
    if dst.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing module: {dst}")
        shutil.rmtree(dst)

    shutil.copytree(src, dst, ignore=ignore_factory(keep_template_hefs))


def copy_hef(hef_source: Path | None, dst_model_dir: Path, hef_name: str) -> None:
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    if hef_source is None:
        readme = dst_model_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "Place the Hailo Model Zoo .hef file for this module in this directory.\n",
                encoding="utf-8",
            )
        return

    source = hef_source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"--hef-source does not exist: {source}")
    if source.suffix.lower() != ".hef":
        raise SystemExit(f"--hef-source must point to a .hef file: {source}")

    shutil.copy2(source, dst_model_dir / hef_name)


def create_dockerfile(template_dockerfile: Path, output_dockerfile: Path, replacements: dict[str, str], force: bool) -> None:
    if output_dockerfile.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing dockerfile: {output_dockerfile}")
        output_dockerfile.unlink()

    text = template_dockerfile.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    output_dockerfile.write_text(text, encoding="utf-8", newline="")


def task_copy(task: str) -> dict[str, str]:
    if task == "segmentation":
        return {
            "en_title": "semantic segmentation",
            "zh_title": "语义分割",
            "en_output": "a per-pixel class mask and per-class area statistics",
            "zh_output": "逐像素类别 mask，以及每个类别的面积统计",
            "en_post": (
                "The segmentation template accepts NHWC softmax outputs and also handles NCHW by "
                "moving the class channel to the end. For other mask layouts, update post_process_hailo()."
            ),
            "zh_post": (
                "分割模板支持 NHWC softmax 输出，也会自动处理 NCHW 通道前置格式。"
                "如果 Hailo Zoo README 里的 mask 布局不同，需要改 post_process_hailo()。"
            ),
            "class_note_en": "For non-VOC datasets, pass --class_path class_config.txt with background as the first class.",
            "class_note_zh": "非 VOC 数据集建议传入 --class_path class_config.txt，第一项应为 background。",
        }

    return {
        "en_title": "object detection",
        "zh_title": "目标检测",
        "en_output": "bounding boxes, class ids, and confidence scores",
        "zh_output": "检测框、类别 id 和置信度",
        "en_post": (
            "The detection template expects a Hailo Model Zoo HEF with on-chip NMS, "
            "usually shaped like (1, num_classes, max_dets, 5). If the HEF has raw heads, "
            "rewrite post_process_hailo()."
        ),
        "zh_post": (
            "检测模板期望使用带片上 NMS 的 Hailo Model Zoo HEF，常见输出接近 "
            "(1, num_classes, max_dets, 5)。如果 HEF 是 raw heads，需要重写 post_process_hailo()。"
        ),
        "class_note_en": "For non-COCO datasets, pass --class_path class_config.txt in model class-index order.",
        "class_note_zh": "非 COCO 数据集建议传入 --class_path class_config.txt，顺序必须和模型类别索引一致。",
    }


def render_readme_en(model: str, task: str, hef_name: str, api_name: str, dockerfile: str, tag: str) -> str:
    copy = task_copy(task)
    title = slug_to_title(model)
    return f"""# {title} on Raspberry Pi 5 + Hailo-8

This module packages the Hailo Model Zoo `{model}` {copy["en_title"]} model as a
one-command Docker deployment for Raspberry Pi 5 / reComputer R20 with Hailo-8.

## Files

| Path | Purpose |
|---|---|
| `web_detection.py` | FastAPI server, MJPEG preview, single-shot inference, and offline video analysis |
| `model/{hef_name}` | Hailo Executable Format model copied from Hailo Model Zoo |
| `hailort-packages/*.whl` | HailoRT Python wheel matching the host driver major.minor version |
| `../../docker/hailo8/{dockerfile}` | Dockerfile for this model image |
| `video/test.mp4` | Built-in demo video |

## Prepare the HEF

Place the Hailo-8 HEF here:

```text
model/{hef_name}
```

The HEF must target `hailo8`. Host `hailo-all`, firmware, mounted
`libhailort.so`, and the wheel in `hailort-packages/` must use matching
HailoRT major.minor versions.

## Build

```bash
cd src/rpi5_hailo8_{model}

sudo docker build -f ../../docker/hailo8/{dockerfile} \\
    -t {tag} .
```

## Run With Demo Video

```bash
sudo docker run --rm --privileged --net=host \\
    -e PYTHONUNBUFFERED=1 \\
    --device /dev/hailo0:/dev/hailo0 \\
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \\
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \\
    {tag}
```

Open:

```text
http://<Pi5_IP>:8000
```

## Run With USB Camera

```bash
sudo docker run --rm --privileged --net=host \\
    -e PYTHONUNBUFFERED=1 \\
    --device /dev/hailo0:/dev/hailo0 \\
    --device /dev/video0:/dev/video0 \\
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \\
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \\
    {tag} \\
    python web_detection.py --model_path model/{hef_name} --camera_id 0
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/{api_name}/predict` | POST | Inference on an uploaded image, a selected video frame, or the current camera frame |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or update confidence / NMS config |
| `/api/video/upload` | POST | Upload a video for offline analysis |
| `/api/video/analyze` | POST | Start offline analysis |
| `/api/video/status` | GET | Poll analysis progress |
| `/api/video/list` | GET | List uploaded and processed videos |
| `/api/video/download/{{filename}}` | GET | Download processed output |

Example:

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/{api_name}/predict -F "file=@test.jpg"
```

The response contains {copy["en_output"]}.

## Classes

{copy["class_note_en"]}

Example `class_config.txt` format:

```text
"background","person","car"
```

Run with custom classes:

```bash
python web_detection.py --model_path model/{hef_name} --video_path video/test.mp4 --class_path class_config.txt
```

## Porting Notes

{copy["en_post"]}

Check the Hailo Model Zoo README for the exact output tensor names, shapes, and
post-processing assumptions before publishing the image.
"""


def render_readme_zh(model: str, task: str, hef_name: str, api_name: str, dockerfile: str, tag: str) -> str:
    copy = task_copy(task)
    title = slug_to_title(model)
    return f"""# {title} on Raspberry Pi 5 + Hailo-8

本模块把 Hailo Model Zoo 的 `{model}` {copy["zh_title"]}模型封装成可在
Raspberry Pi 5 / reComputer R20 + Hailo-8 上一键部署的 Docker 镜像。

## 文件说明

| 路径 | 作用 |
|---|---|
| `web_detection.py` | FastAPI 服务、MJPEG 预览、单帧推理、离线视频分析 |
| `model/{hef_name}` | 从 Hailo Model Zoo 获取的 Hailo Executable Format 模型 |
| `hailort-packages/*.whl` | 与宿主机驱动 major.minor 匹配的 HailoRT Python wheel |
| `../../docker/hailo8/{dockerfile}` | 该模型对应的 Dockerfile |
| `video/test.mp4` | 内置测试视频 |

## 准备 HEF

把面向 `hailo8` 的模型放到：

```text
model/{hef_name}
```

宿主机 `hailo-all`、固件、bind mount 的 `libhailort.so`、以及
`hailort-packages/` 里的 wheel 必须使用匹配的 HailoRT major.minor 版本。

## 构建镜像

```bash
cd src/rpi5_hailo8_{model}

sudo docker build -f ../../docker/hailo8/{dockerfile} \\
    -t {tag} .
```

## 使用内置视频运行

```bash
sudo docker run --rm --privileged --net=host \\
    -e PYTHONUNBUFFERED=1 \\
    --device /dev/hailo0:/dev/hailo0 \\
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \\
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \\
    {tag}
```

浏览器打开：

```text
http://<Pi5_IP>:8000
```

## 使用 USB 摄像头运行

```bash
sudo docker run --rm --privileged --net=host \\
    -e PYTHONUNBUFFERED=1 \\
    --device /dev/hailo0:/dev/hailo0 \\
    --device /dev/video0:/dev/video0 \\
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \\
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \\
    {tag} \\
    python web_detection.py --model_path model/{hef_name} --camera_id 0
```

## API

| Endpoint | 方法 | 说明 |
|---|---|---|
| `/api/models/{api_name}/predict` | POST | 对上传图片、视频指定帧或摄像头当前帧做一次推理 |
| `/api/video_feed` | GET | MJPEG 实时预览流 |
| `/api/config` | GET / POST | 读取或更新置信度 / NMS 配置 |
| `/api/video/upload` | POST | 上传视频用于离线分析 |
| `/api/video/analyze` | POST | 启动离线分析 |
| `/api/video/status` | GET | 轮询分析进度 |
| `/api/video/list` | GET | 列出上传和输出文件 |
| `/api/video/download/{{filename}}` | GET | 下载处理结果 |

调用示例：

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/{api_name}/predict -F "file=@test.jpg"
```

响应中会包含{copy["zh_output"]}。

## 类别配置

{copy["class_note_zh"]}

`class_config.txt` 示例：

```text
"background","person","car"
```

带自定义类别运行：

```bash
python web_detection.py --model_path model/{hef_name} --video_path video/test.mp4 --class_path class_config.txt
```

## 迁移检查

{copy["zh_post"]}

发布镜像前，建议对照 Hailo Model Zoo 里该模型的 README，确认输出张量名称、shape
和后处理假设都一致。
"""


def write_generated_readmes(module_dir: Path, model: str, task: str, hef_name: str, api_name: str, dockerfile: str, tag: str) -> None:
    (module_dir / "README.md").write_text(
        render_readme_en(model, task, hef_name, api_name, dockerfile, tag),
        encoding="utf-8",
        newline="",
    )
    (module_dir / "README_zh.md").write_text(
        render_readme_zh(model, task, hef_name, api_name, dockerfile, tag),
        encoding="utf-8",
        newline="",
    )


def infer_task(template_name: str, fallback: str) -> str:
    for task, default_template in TEMPLATES.items():
        if default_template == template_name:
            return task
    return fallback


def main() -> int:
    args = parse_args()
    model = kebab_to_snake(args.model)
    api_name = args.api_name or model
    hef_name = args.hef_name or (args.hef_source.name if args.hef_source else f"{model}.hef")
    if not hef_name.endswith(".hef"):
        hef_name = f"{hef_name}.hef"

    template_name = args.template or TEMPLATES[args.task]
    readme_task = infer_task(template_name, args.task)
    template_dir = SRC_DIR / template_name
    if not template_dir.exists():
        raise SystemExit(f"Template module not found: {template_dir}")

    defaults = TEMPLATE_DEFAULTS.get(template_name)
    if defaults is None:
        raise SystemExit(
            f"No defaults known for template {template_name!r}. "
            "Add it to TEMPLATE_DEFAULTS before scaffolding from it."
        )

    module_dir = SRC_DIR / f"rpi5_hailo8_{model}"
    dockerfile = DOCKER_DIR / f"{model}.dockerfile"
    template_dockerfile = DOCKER_DIR / defaults["dockerfile"]
    if not template_dockerfile.exists():
        raise SystemExit(f"Template dockerfile not found: {template_dockerfile}")

    default_hef_stem = Path(defaults["hef"]).stem
    new_hef_stem = Path(hef_name).stem
    default_dockerfile = defaults["dockerfile"]
    new_dockerfile = f"{model}.dockerfile"
    old_title_slug = defaults["api"]
    old_readme_tag = (
        "rpi5-hailo8-segformer"
        if template_name == "rpi5_hailo8_segformer_b0_bn"
        else f"rpi5-hailo8-{old_title_slug}"
    )
    new_readme_tag = f"r20-hailo8-{model}"

    broad_replacements = {
        defaults["hef"]: hef_name,
        default_hef_stem: new_hef_stem,
        default_dockerfile: new_dockerfile,
        old_readme_tag: new_readme_tag,
        defaults["title"]: f"{slug_to_title(model)} on RPi5 + Hailo-8",
        template_name: f"rpi5_hailo8_{model}",
    }
    text_replacements = {
        **broad_replacements,
        f"/api/models/{defaults['api']}/predict": f"/api/models/{api_name}/predict",
        f"api/models/{defaults['api']}/predict": f"api/models/{api_name}/predict",
        f"/api/models/yolov5/": f"/api/models/{api_name}/",
        f"--model_path {hef_name}": f"--model_path model/{hef_name}",
    }

    copy_template(template_dir, module_dir, args.keep_template_hefs, args.force)
    copy_hef(args.hef_source, module_dir / "model", hef_name)
    create_dockerfile(template_dockerfile, dockerfile, broad_replacements, args.force)

    tag = f"{args.image_prefix}-{model}:latest"
    rel_module = module_dir.relative_to(ROOT).as_posix()
    rel_dockerfile = dockerfile.relative_to(ROOT).as_posix()

    for path in (
        module_dir / "web_detection.py",
        module_dir / "TEST_REPORT.md",
        module_dir / "model" / "README.md",
    ):
        if path.exists():
            replace_text(path, text_replacements)

    if args.keep_template_readmes:
        for path in (module_dir / "README.md", module_dir / "README_zh.md"):
            if path.exists():
                replace_text(path, text_replacements)
    else:
        write_generated_readmes(
            module_dir=module_dir,
            model=model,
            task=readme_task,
            hef_name=hef_name,
            api_name=api_name,
            dockerfile=new_dockerfile,
            tag=tag,
        )

    print(f"Created module: {rel_module}")
    print(f"Created dockerfile: {rel_dockerfile}")
    print(
        "Generated READMEs: "
        f"{rel_module}/README.md, {rel_module}/README_zh.md"
        if not args.keep_template_readmes
        else "Kept template README.md / README_zh.md with replacements"
    )
    if args.hef_source:
        print(f"Copied HEF: {rel_module}/model/{hef_name}")
    else:
        print(f"Next: place the Hailo Model Zoo HEF at {rel_module}/model/{hef_name}")
    print()
    print("Build on the Raspberry Pi:")
    print(f"  cd {rel_module}")
    print(f"  sudo docker build -f ../../{rel_dockerfile} -t {tag} .")
    print()
    print("Run:")
    print("  sudo docker run --rm --privileged --net=host \\")
    print("      -e PYTHONUNBUFFERED=1 \\")
    print("      --device /dev/hailo0:/dev/hailo0 \\")
    print("      -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \\")
    print("      -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \\")
    print(f"      {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
