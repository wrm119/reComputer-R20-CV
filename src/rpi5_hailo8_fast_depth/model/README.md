# FastDepth HEF

This directory contains the Hailo-8 HEF used by the depth-estimation module:

```text
fast_depth.hef
```

`../web_detection.py` reads the HEF input size at startup, decodes the raw output
to a relative-depth map, restores the map to the original frame coordinates, and
renders a pseudo-color depth overlay.
