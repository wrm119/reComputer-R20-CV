# SCRFD HEF models

This directory contains the SCRFD face-detection HEFs used by the Docker image:

```text
scrfd_500m.hef
scrfd_2.5g.hef
scrfd_10g.hef
```

All three models are expected to use 640x640x3 input and target Hailo-8.

The default Docker command uses:

```text
model/scrfd_500m.hef
```

To switch models, override the container command:

```bash
python web_detection.py --model_path model/scrfd_2.5g.hef --video_path video/test.mp4
python web_detection.py --model_path model/scrfd_10g.hef --video_path video/test.mp4
```

`../web_detection.py` implements SCRFD raw score / bbox / landmark decoding,
confidence filtering, NMS, and 5-point landmark drawing.
