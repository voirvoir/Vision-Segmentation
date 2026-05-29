from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from vision_backend.detector import DetectionResult
from vision_backend.protocol import Detection


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detect_image.py"
spec = importlib.util.spec_from_file_location("detect_image_script", SCRIPT_PATH)
detect_image_script = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(detect_image_script)


class DetectImageScriptTests(unittest.TestCase):
    def test_draw_detections_changes_pixels_on_box(self):
        image = Image.new("RGB", (100, 80), color=(255, 255, 255))
        detections = (
            Detection(
                class_id=0,
                label="person",
                confidence=0.9,
                bbox_xyxy=(10.0, 10.0, 60.0, 60.0),
                mask_polygon_xy=((10.0, 10.0), (60.0, 10.0), (60.0, 60.0)),
            ),
        )

        annotated = detect_image_script.draw_detections(image, detections)

        self.assertNotEqual(annotated.getpixel((10, 10)), (255, 255, 255))

    def test_write_json_response_can_write_server_json_line(self):
        detection_result = DetectionResult(
            image_width=100,
            image_height=80,
            model="fake-model.pt",
            detections=(
                Detection(
                    class_id=1,
                    label="cup",
                    confidence=0.8,
                    bbox_xyxy=(1.0, 2.0, 30.0, 40.0),
                    mask_polygon_xy=((1.0, 2.0), (30.0, 2.0), (30.0, 40.0)),
                ),
            ),
            latency_ms=3.456,
        )
        response = detect_image_script.result_to_vision_response(
            frame_id=9,
            bytes_received=1234,
            detection_result=detection_result,
        )

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sample.detections.json"
            detect_image_script.write_json_response(output_path, response, pretty=False)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["frame_id"], 9)
        self.assertEqual(payload["bytes_received"], 1234)
        self.assertEqual(payload["image_width"], 100)
        self.assertEqual(payload["image_height"], 80)
        self.assertEqual(payload["model"], "fake-model.pt")
        self.assertEqual(payload["latency_ms"], 3.46)
        self.assertEqual(payload["detections"][0]["label"], "cup")
        self.assertEqual(payload["detections"][0]["mask_polygon_xy"], [[1.0, 2.0], [30.0, 2.0], [30.0, 40.0]])


if __name__ == "__main__":
    unittest.main()
