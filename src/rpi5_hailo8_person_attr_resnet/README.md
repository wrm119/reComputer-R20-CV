# Person Attribute ResNet on Raspberry Pi 5 + Hailo-8

This module packages `person_attr_resnet_v1_18.hef` as a one-command Docker
deployment for Raspberry Pi 5 / CM5 with Hailo-8.

The model is a person-attribute classifier. It expects a cropped person image
or a frame where the target person is dominant, then predicts PETA attributes
such as age range, gender, hat, logo, long hair, muffler, plastic bag, and
sunglasses.

## Model

- Task: Person Attribute
- Input shape: `224x224x3`
- Output: 35 PETA attribute logits / scores
- Postprocess: sigmoid, then threshold. Default threshold is `0.70`, matching
  the Hailo TAPPAS person-attributes postprocess.
- REST API route: `/api/models/person_attr_resnet/predict`

## Prerequisites

### Install Docker

Run the following commands on the development board to install Docker:

```bash
# Download installation script
curl -fsSL https://get.docker.com -o get-docker.sh
# Install using Aliyun mirror source
sudo sh get-docker.sh --mirror Aliyun
# Start Docker and enable auto-start on boot
sudo systemctl enable docker
sudo systemctl start docker
```

## Build

```bash
cd src/rpi5_hailo8_person_attr_resnet

sudo docker build -f ../../docker/hailo8/person_attr_resnet.dockerfile \
    -t r20-hailo8-person_attr_resnet:latest .
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-person_attr_resnet:latest
```

Open:

```text
http://<Pi5_IP>:8000
```

## Camera

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    r20-hailo8-person_attr_resnet:latest \
    python web_detection.py --model_path model/person_attr_resnet_v1_18.hef --camera_id 0
```

## API

```bash
curl -X POST http://<Pi5_IP>:8000/api/models/person_attr_resnet/predict \
    -F "file=@person.jpg"
```

The response includes:

- `predictions`: filtered attribute list
- `attributes`: same attribute list, kept for clarity
- `threshold`: active threshold
- `image`: base64 JPEG with attributes drawn on the image

## Notes

This model is not a detector. For best results, provide a person crop or a
frame where the person occupies the main visual area. If the final product
needs per-person attributes in a multi-person scene, run a person detector
first and send each person crop into this module.
