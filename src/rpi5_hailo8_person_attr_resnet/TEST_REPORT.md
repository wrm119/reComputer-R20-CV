# Person Attribute ResNet Validation Checklist

## Local Checks

- `web_detection.py` compiles with `python -m py_compile`.
- Dockerfile default command points to `model/person_attr_resnet_v1_18.hef`.
- REST API route is `/api/models/person_attr_resnet/predict`.

## On-device Checks

1. Build `r20-hailo8-person_attr_resnet:latest`.
2. Run the container on Raspberry Pi 5 / CM5 with Hailo-8.
3. Open `http://<Pi5_IP>:8000`.
4. Confirm the first inference prints the raw output shape and decoded attribute stats.
5. Move the threshold slider and confirm attributes update.
6. Test `/api/models/person_attr_resnet/predict` with a person crop.

Expected first-inference log:

```text
[PersonAttrResNet] output=..., raw shape=..., decoded attrs=..., min=..., max=...
```
