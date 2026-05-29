# Vision Segmentation Project Progress

Last updated: May 5, 2026

## Goal

Build an end-to-end VR vision MVP:

- Unity/Quest captures passthrough camera frames.
- Unity sends each encoded image to the Python backend over TCP.
- The Python backend runs YOLO segmentation/detection.
- The backend returns JSON detections with image-space bounding boxes and optional mask polygons.
- Unity parses the JSON and draws 2D bounding boxes over the camera preview.

## Repositories and Key Paths

- Backend repo: `D:\Documents 2TB\MSU Classes\CSE 422\Honors Option\Vision-Segmentation`
- Unity client repo: `D:\Projects\Unity\Unity-PassthroughCameraApiSamples`
- Backend server: `src/vision_backend/server.py`
- Backend detector: `src/vision_backend/detector.py`
- Backend protocol: `src/vision_backend/protocol.py`
- Unity client: `Assets/PassthroughCameraApiSamples/CameraViewer/Scripts/CameraViewerManager.cs`
- Unity prefab: `Assets/PassthroughCameraApiSamples/CameraViewer/Prefabs/CameraViewerManagerPrefab.prefab`

## Backend Status

- TCP protocol is still 4-byte little-endian payload length followed by encoded image bytes.
- Server returns one newline-terminated JSON response per frame.
- Model is loaded once at server startup, not per frame.
- Default segmentation model is `yolo26s-seg.pt`.
- GPU is used when `--device auto` resolves to CUDA.
- Debug frame saving is preserved, but live VR should use `--no-frame-log`.
- Inference is run off the asyncio event loop and guarded by one shared inference lock.
- Response JSON includes:
  - `frame_id`
  - `status`
  - `bytes_received`
  - `image_width`
  - `image_height`
  - `model`
  - `detections`
  - `bbox_xyxy`
  - optional `mask_polygon_xy`
  - `latency_ms`

## Unity Client Status

- Unity sends frames using the existing TCP request/response protocol.
- Backend endpoint parsing uses the Inspector/prefab endpoint instead of a hardcoded host.
- Response size cap is raised to `262144` bytes.
- JSON parsing uses Newtonsoft DTOs for full server responses.
- Server error responses are parsed and shown as backend errors.
- Client keeps one request in flight.
- TCP connection is persistent across frames and closes on stop or socket failure.
- Capture path now uses `AsyncGPUReadback`.
- JPEG encoding uses background `Task.Run` with `ImageConversion.EncodeArrayToJPG`.
- Bounding boxes are drawn as pooled UI overlay objects under the preview `RawImage`.
- Overlay supports:
  - box pooling
  - configurable border thickness
  - optional high-contrast color
  - debug fixed box mode
  - mirror X/Y calibration
  - 0/90/180/270 rotation calibration
  - aspect-ratio-aware placement
- Overlay debug text reports:
  - detections received
  - boxes attempted
  - boxes visible
  - overlay pool size
  - active box count

## Useful Commands

Run live backend:

```powershell
cd "D:\Documents 2TB\MSU Classes\CSE 422\Honors Option\Vision-Segmentation"
uv run vision-server --host 0.0.0.0 --port 40081 --model yolo26s-seg.pt --device auto --no-frame-log
```

Verify PyTorch/CUDA:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run backend tests:

```powershell
uv run python -m unittest discover -s tests
```

Run one saved-frame detection test:

```powershell
uv run python scripts/detect_image.py logs/frames/<frame-file>.jpg --pretty-json
```

Benchmark saved frames:

```powershell
uv run python scripts/benchmark_detector.py --model yolo26s-seg.pt --model yolo26n-seg.pt --frames logs/frames
```

## Verification So Far

- Backend unit tests passed with `uv run python -m unittest discover -s tests`.
- Unity batchmode compile/import check exited with code `0`.
- Unity log search found no C# compiler errors in `CameraViewerManager.cs`.
- `git diff --check` passed for edited Unity files.
- Direct Roslyn compile of the whole Unity project found an unrelated pre-existing issue in `SentisInferenceRunManager.cs`, not in `CameraViewerManager.cs`.

## Current VR Test Checklist

1. Start the backend with `--no-frame-log`.
2. Confirm Unity prefab endpoint points to the PC LAN IP, for example `192.168.50.143:40081`.
3. Run the headset for 10-20 frames first.
4. Confirm server logs show one connection and increasing `frame_id` values.
5. Confirm Unity debug text shows timing and overlay stats.
6. Confirm `Detections > 0` when an object is visible.
7. Confirm boxes appear over the preview.
8. If detections appear but boxes do not, enable `m_drawDebugBox`.
9. If the debug box appears but detection boxes do not, inspect `bbox_xyxy` parsing.
10. If boxes are offset/flipped/rotated, tune overlay mirror/rotation settings in the prefab.
11. After the smoke test passes, run a 100-frame test and watch for stutter, response-size errors, and object count growth.

## Known Risks and Next Fixes

- Server still returns mask polygons by default, so a crowded scene can still exceed Unity's `262144` byte response cap.
- Next backend hardening should add a live boxes-only response mode or cap/simplify mask polygons.
- Overlay alignment is not fully proven until tested in headset.
- Async GPU readback and JPEG encoding compile, but Quest runtime behavior still needs live validation.
- If stop is pressed while a socket operation is in flight, Unity may log a harmless socket error as the stream closes.

## Recommended Next Milestone

Run the VR smoke test now. If the smoke test passes, do the 100-frame stability test. If response size or stale/misaligned boxes appear, prioritize:

1. Backend boxes-only live response mode.
2. Overlay calibration defaults for Quest preview orientation.
3. Stale-result suppression based on frame age.
