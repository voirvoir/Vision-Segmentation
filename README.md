# Vision Backend

TCP backend for a Meta Quest / Unity client that streams encoded camera frames to a
remote Python process. The server decodes each frame, runs GPU-backed instance
segmentation, and returns image-space boxes plus mask polygons that Unity can draw.

## What it does

- Accepts TCP connections on `0.0.0.0:40081` by default.
- Receives frames using the Unity-compatible protocol:
  - 4-byte little-endian frame length (`Int32`)
  - encoded image bytes, such as JPEG or PNG
- Optionally saves debug copies to `logs/frames`.
- Runs a YOLO segmentation model once per received frame.
- Returns one UTF-8 JSON line per frame.

Example response:

```json
{"frame_id":1,"status":"ok","bytes_received":54044,"image_width":640,"image_height":480,"model":"yolo26s-seg.pt","detections":[{"class_id":0,"label":"person","confidence":0.91,"bbox_xyxy":[120.5,44.0,301.2,420.7],"mask_polygon_xy":[[121.0,45.0],[300.0,45.0],[301.0,419.0]]}],"latency_ms":18.4}
```

Error responses are also JSON lines:

```json
{"frame_id":1,"status":"error","bytes_received":1024,"error":"invalid_image","message":"Frame payload is not a decodable image"}
```

## Project layout

- `src/vision_backend/server.py`: async multi-client TCP server
- `src/vision_backend/detector.py`: image decoding and Ultralytics detector adapter
- `src/vision_backend/protocol.py`: frame header helpers and JSON response models
- `src/vision_backend/frame_logger.py`: debug frame dump and JPEG conversion
- `scripts/send_test_frame.py`: local sender harness
- `scripts/benchmark_detector.py`: saved-frame model benchmark harness
- `tests/`: unit and async integration tests

## Setup with uv

```powershell
cd "d:\Documents 2TB\MSU Classes\CSE 422\Honors Option\Vision-Segmentation"
uv venv
uv sync
```

Install detection dependencies:

```powershell
uv sync --extra detection
```

Verify PyTorch can see the GPU:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If that prints `False` on an NVIDIA machine, install the CUDA-enabled PyTorch
build. With `uv`, force the PyTorch packages onto the CUDA backend:

```powershell
uv pip install torch torchvision torchaudio --torch-backend cu128 --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```

References:

- https://docs.ultralytics.com/models/yolo26/
- https://docs.ultralytics.com/compare/
- https://pytorch.org/get-started/locally/

## Run the server

```powershell
uv run vision-server
```

Recommended live VR run:

```powershell
uv run vision-server --host 0.0.0.0 --port 40081 --model yolo26s-seg.pt --device auto --no-frame-log
```

Useful options:

```powershell
uv run vision-server --model yolo26n-seg.pt --conf-threshold 0.25 --max-detections 50 --imgsz 640
uv run vision-server --frame-log-dir logs/frames --log-level DEBUG
```

`yolo26s-seg.pt` is the default balanced segmentation model. Use
`yolo26n-seg.pt` when latency matters more than mask quality.

## Send a local test frame

Run in another terminal while the server is running:

```powershell
uv run python scripts/send_test_frame.py --host 127.0.0.1 --port 40081
```

Send a saved VR frame:

```powershell
uv run python scripts/send_test_frame.py --image logs/frames/example.jpg
```

Send invalid bytes to test error handling:

```powershell
uv run python scripts/send_test_frame.py --raw-size 2048
```

## Benchmark saved frames

```powershell
uv run python scripts/benchmark_detector.py --frames logs/frames
```

Compare specific models:

```powershell
uv run python scripts/benchmark_detector.py --model yolo26s-seg.pt --model yolo26n-seg.pt --frames logs/frames
```

## Detect one image offline

Use this to verify the JSON response and expected bounding-box overlay before
connecting Unity:

```powershell
uv run python scripts/detect_image.py logs/frames/example.jpg --pretty-json
```

By default it writes:

- `outputs/detection_tests/<image>.detections.json`
- `outputs/detection_tests/<image>.annotated.jpg`

## Run tests

```powershell
uv run python -m unittest discover -s tests -p "test_*.py"
```

The default tests use fake detectors and do not download model weights.

## Unity compatibility notes

- Unity `BitConverter.GetBytes(imageBytes.Length)` sends little-endian `Int32`,
  which this server expects.
- Send encoded image bytes, for example `Texture2D.EncodeToJPG()` or
  `Texture2D.EncodeToPNG()`.
- Returned boxes and mask polygons are pixel coordinates in the original decoded image space:
  `[x_min, y_min, x_max, y_max]`.
- Parse responses as newline-terminated JSON objects.
