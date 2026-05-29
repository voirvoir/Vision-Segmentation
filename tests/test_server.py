from __future__ import annotations

import asyncio
from io import BytesIO
import json
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from vision_backend.detector import DetectionResult
from vision_backend.protocol import Detection, pack_frame_length
from vision_backend.server import handle_client


class FakeDetector:
    model_name = "fake-model.pt"

    def detect(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(
            image_width=image.width,
            image_height=image.height,
            model=self.model_name,
            detections=(
                Detection(
                    class_id=0,
                    label="person",
                    confidence=0.9,
                    bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
                    mask_polygon_xy=((10.0, 20.0), (100.0, 20.0), (100.0, 200.0)),
                ),
            ),
            latency_ms=2.25,
        )


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def _send_frame(self, payload: bytes) -> dict[str, object]:
        with TemporaryDirectory() as tmp_dir:
            server = await asyncio.start_server(
                lambda reader, writer: handle_client(
                    reader,
                    writer,
                    max_frame_bytes=10 * 1024 * 1024,
                    frame_log_dir=tmp_dir,
                    detector=FakeDetector(),
                    inference_lock=asyncio.Lock(),
                    save_debug_frames=False,
                ),
                "127.0.0.1",
                0,
            )

            host, port = server.sockets[0].getsockname()[:2]
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.write(pack_frame_length(len(payload)))
                writer.write(payload)
                await writer.drain()

                response = await reader.readline()
                writer.close()
                await writer.wait_closed()
            finally:
                server.close()
                await server.wait_closed()

        return json.loads(response)

    async def test_server_returns_detection_response_for_image(self):
        buffer = BytesIO()
        Image.new("RGB", (640, 480), color=(40, 100, 180)).save(buffer, format="JPEG")

        payload = await self._send_frame(buffer.getvalue())

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["image_width"], 640)
        self.assertEqual(payload["image_height"], 480)
        self.assertEqual(payload["model"], "fake-model.pt")
        self.assertEqual(payload["latency_ms"], 2.25)
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["detections"][0]["bbox_xyxy"], [10.0, 20.0, 100.0, 200.0])
        self.assertEqual(
            payload["detections"][0]["mask_polygon_xy"],
            [[10.0, 20.0], [100.0, 20.0], [100.0, 200.0]],
        )

    async def test_server_returns_error_for_invalid_image(self):
        payload = await self._send_frame(b"not-an-image")

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "invalid_image")
        self.assertEqual(payload["message"], "Frame payload is not a decodable image")


if __name__ == "__main__":
    unittest.main()
