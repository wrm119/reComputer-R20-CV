# Base image for arm64 (Raspberry Pi 5). Build on the Pi directly, or use
# `docker buildx build --platform linux/arm64` from another host.
FROM python:3.11-slim

WORKDIR /app

# OpenCV + Hailo runtime dependencies.
# libgl1: cv2 image codecs
# libglib2.0-0, libsm6, libxext6, libxrender1: common cv2 dependencies
# libgomp1: OpenMP (used by some opencv-python ops)
# ffmpeg: standalone binary used by VideoAnalyzer for libx264 ultrafast
#         encoding (~5x faster than cv2's mp4v at 4K on Pi 5).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the local HailoRT wheel. The user must place a wheel matching their
# host driver version into hailort-packages/ before building this image.
# Example filename: hailort-4.23.0-cp311-cp311-linux_aarch64.whl
#
# build-essential + python3-dev are needed to compile `netifaces`, a hailort
# dependency that has no pre-built wheel for cp311/aarch64. They are removed
# after install to keep the final image small.
COPY hailort-packages/*.whl /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev \
    && pip install --no-cache-dir /tmp/hailort-*.whl \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /tmp/*.whl

COPY . .

EXPOSE 8000

CMD ["python", "web_detection.py", "--model_path", "model/scdepthv3.hef", "--video_path", "video/test.mp4"]
