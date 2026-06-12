# HailoRT Python wheel directory

Place a `hailort-<version>-cp311-cp311-linux_aarch64.whl` here. The Dockerfile
installs whatever it finds matching `hailort-*.whl`.

## Where to get the wheel

1. Register at the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
2. Go to **Software Downloads → HailoRT**
3. Pick the version whose **major.minor** matches the firmware version reported by
   `hailortcli fw-control identify` on the host. Mismatched majors WILL fail at
   runtime with a firmware version error.
4. Download the **Python wheel** (not the `.deb` / `.tar.gz`) for:
   - Architecture: `aarch64` (Raspberry Pi 5 is arm64)
   - Python: `cp311` (the Dockerfile uses `python:3.11-slim`)
5. Drop the `.whl` into this directory:

```
src/rpi5_hailo8_yolov5/hailort-packages/hailort-4.23.0-cp311-cp311-linux_aarch64.whl
```

The exact filename will vary by version — just make sure the wildcard
`hailort-*.whl` matches exactly one file (the Dockerfile's `pip install
/tmp/hailort-*.whl` step assumes a single wheel).

## Bare-metal install (no Docker)

Same wheel:

```bash
pip install hailort-packages/hailort-*.whl
```
