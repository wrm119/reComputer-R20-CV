# Hailo-8 YOLOv5 联机测试报告

**状态**：待补 — 本模块尚未做端到端验证。

---

## 待办

按以下顺序在 Pi 5 + Hailo-8 上跑通后再补：

1. 构建镜像
   ```bash
   cd src/rpi5_hailo8_yolov5
   sudo docker build -f ../../docker/hailo8/yolov5.dockerfile -t rpi5-hailo8-yolov5:latest .
   ```
2. 容器启动 + `GET /` + `GET /api/config`
3. 单帧推理 `POST /api/models/yolov5/predict`
4. MJPEG 视频流 `GET /api/video_feed`
5. 视频上传 + 异步分析（`/api/video/upload` → `/api/video/analyze` → `/api/video/status` → `/api/video/download`）
6. 18 分钟以上压测，扫日志确认无 error/warning/exception

完整流程可参考姐妹模块 [yolov8 的测试报告](../rpi5_hailo8_yolov8/TEST_REPORT.md)，重点关注：

- YOLOv5 Model Zoo `.hef` 输出布局是否与 v8 一致（NMS-on-chip 且 `(1, num_classes, max_dets, 5)`）。如果不一致，需要按 [README.md § 7](README.md#7-adapting-to-other-models) 重写 `post_process_hailo()`。
- HailoRT 版本一致性：宿主机固件 / `libhailort.so` / 容器内 wheel 三处的 major.minor 必须对齐。
