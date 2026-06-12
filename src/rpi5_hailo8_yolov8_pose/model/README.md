# YOLOv8 Pose HEFs

This directory contains the Hailo-8 HEF files used by the pose module:

| File | Notes |
|---|---|
| `yolov8s_pose.hef` | Default model used by the Docker CMD |
| `yolov8m_pose.hef` | Larger model for higher accuracy testing |

Both models are expected to use input shape `640x640x3`.

`../web_detection.py` implements person box decoding, COCO 17-keypoint decoding,
letterbox coordinate restoration, and skeleton drawing.
