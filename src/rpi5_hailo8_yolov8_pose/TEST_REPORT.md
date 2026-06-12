# YOLOv8 Pose Validation Checklist

This module has been scaffolded and statically checked on the development
machine. Full inference must be validated on Raspberry Pi 5 / reComputer R20
with Hailo-8 hardware.

## Static Checks

- `web_detection.py` compiles with Python.
- Dockerfile default command points to `model/yolov8s_pose.hef`.
- Both HEFs are present in `model/`.
- REST API route is `/api/models/yolov8_pose/predict`.

## Hardware Checks

1. Build `r20-hailo8-yolov8_pose:latest`.
2. Run the default demo video and open `http://<Pi5_IP>:8000`.
3. Confirm the startup log prints the YOLOv8 Pose output type and shape.
4. Confirm person boxes appear on people.
5. Confirm COCO 17 keypoints and skeleton lines align with the body.
6. Test `model/yolov8m_pose.hef` for accuracy and latency comparison.
7. Test USB camera mode and verify the preview does not accumulate stale frames.

If boxes or keypoints are transposed, capture this log line:

```text
[YOLOv8 Pose] raw output type=..., shape=...
```

The post-process layout can then be adjusted against the actual HailoRT output.
