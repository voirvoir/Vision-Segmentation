# Vision Backend (TCP Ingestion Server)

This backend is the first step of your distributed Meta Quest 3 + remote GPU architecture.
For now, it focuses on reliable TCP frame ingestion and acknowledgement.

## What it does now

- Accepts TCP connections (default `0.0.0.0:40081`)
- Receives frame bytes using your Unity client protocol:
  - 4-byte little-endian header (`Int32`) for frame size
  - raw image/frame payload bytes
- Dumps received frames into `logs/frames` (tries to decode and save as `.jpg`)
- Returns a UTF-8 JSON line response per frame, for example:
  - `{"frame_id":1,"bytes_received":1024,"status":"ok"}`

## Project layout

- `src/vision_backend/server.py`: async multi-client TCP server
- `src/vision_backend/protocol.py`: protocol helpers and response model
- `src/vision_backend/frame_logger.py`: debug frame dump and JPEG conversion
- `scripts/send_test_frame.py`: simple local sender harness
- `tests/test_protocol.py`: minimal protocol unit tests

## Setup with uv

```powershell
cd d:\Main\CodeHub\VisionSegmentation
uv venv
uv sync
```

## Run the server

```powershell
uv run vision-server
```

Optional custom host/port:

```powershell
uv run vision-server --host 0.0.0.0 --port 40081 --log-level DEBUG
```

Optional custom log directory:

```powershell
uv run vision-server --frame-log-dir logs/frames
```

## Send a local test frame

Run in another terminal while server is running:

```powershell
uv run python scripts/send_test_frame.py --host 127.0.0.1 --port 40081 --size 2048
```

## Run tests

```powershell
uv run python -m unittest discover -s tests -p "test_*.py"
```

## Notes for Unity client compatibility

Your Unity `BitConverter.GetBytes(imageBytes.Length)` sends little-endian `Int32`, which this server expects.

To get a viewable `.jpg` in logs, `imageBytes` should be encoded image data (for example `Texture2D.EncodeToJPG()` or `Texture2D.EncodeToPNG()`).
If non-image bytes are sent, the server stores them as `.bin` for debugging.

## Next suggested steps

1. Add frame metadata (timestamp, camera ID, sequence number).
2. Add binary response framing for robust parsing on Unity side.
3. Add model inference pipeline (GPU server) behind a queue.
4. Add compression and authentication for production Wi-Fi use.
