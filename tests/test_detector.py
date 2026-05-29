from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image

from vision_backend.detector import DetectionResult, decode_image
from vision_backend.protocol import Detection


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
                    bbox_xyxy=(1.0, 2.0, 10.0, 20.0),
                    mask_polygon_xy=((1.0, 2.0), (10.0, 2.0), (10.0, 20.0)),
                ),
            ),
            latency_ms=1.5,
        )


class DetectorTests(unittest.TestCase):
    def test_decode_image_returns_rgb_image(self):
        buffer = BytesIO()
        Image.new("RGB", (16, 8), color=(255, 0, 0)).save(buffer, format="PNG")

        image = decode_image(buffer.getvalue())

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (16, 8))

    def test_decode_image_rejects_non_image_bytes(self):
        with self.assertRaises(ValueError):
            decode_image(b"not-an-image")

    def test_fake_detector_result_uses_input_image_size(self):
        image = Image.new("RGB", (640, 480))
        detector = FakeDetector()

        result = detector.detect(image)

        self.assertEqual(result.image_width, 640)
        self.assertEqual(result.image_height, 480)
        self.assertEqual(result.model, "fake-model.pt")
        self.assertEqual(len(result.detections), 1)
        self.assertIsNotNone(result.detections[0].mask_polygon_xy)


if __name__ == "__main__":
    unittest.main()
