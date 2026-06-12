# FastDepth Validation Checklist

This module has been scaffolded and statically checked on the development
machine. Full inference must be validated on Raspberry Pi 5 / reComputer R20
with Hailo-8 hardware.

## Static Checks

- `web_detection.py` compiles with Python.
- Dockerfile default command points to `model/fast_depth.hef`.
- REST API route is `/api/models/fast_depth/predict`.
- The API returns a YOLO-style `predictions` list plus a `depth` statistics object.

## Hardware Checks

1. Build `r20-hailo8-fast_depth:latest`.
2. Run the default demo video and open `http://<Pi5_IP>:8000`.
3. Confirm the startup log prints the HEF input size.
4. Confirm the first inference prints the FastDepth raw output shape and depth stats.
5. Confirm the preview displays a pseudo-color depth result.
6. Test `/api/models/fast_depth/predict` with an uploaded image.
7. Test USB camera mode and verify the preview does not accumulate stale frames.

If the depth map is blank, inverted, or stretched, capture this log line:

```text
[FastDepth] raw output shape=..., decoded depth shape=..., min=..., max=...
```

The post-process layout can then be adjusted against the actual HailoRT output.
