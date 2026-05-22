# Model directory

Place a Hailo `.hef` model here. The default container CMD expects `yolov8n.hef`.

## Where to get yolov8n.hef

1. Visit the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
2. Find the YOLOv8 nano entry under the "Compiled" section for **Hailo-8** (not
   Hailo-8L or Hailo-15)
3. Download the `.hef` file whose HailoRT version line matches your installed
   HailoRT/driver version (check with `hailortcli fw-control identify`)
4. Drop the file into this directory:

```
src/rpi5_hailo8_yolov8/model/yolov8n.hef
```

## Format expectations

`post_process_hailo()` in [../web_detection.py](../web_detection.py) assumes the
`.hef` includes the **NMS post-process layer** (the Model Zoo default). The output
is one tensor of shape `(batch, num_classes, max_dets, 5)` or an equivalent object
array, where each detection is `[ymin, xmin, ymax, xmax, score]` normalized to
`[0, 1]` relative to the network input.

If you compile your own `.hef` without that layer, you'll get raw multi-branch
tensors that need DFL + box decoding — see the original RKNN reference's
`post_process_with_thresh()` for the algorithm, then re-implement against the
Hailo output names.
