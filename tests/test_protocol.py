import json
import unittest

from vision_backend.protocol import (
    HEADER_SIZE,
    Detection,
    VisionResult,
    pack_frame_length,
    unpack_frame_length,
)


class ProtocolTests(unittest.TestCase):
    def test_length_header_roundtrip(self):
        payload_len = 123456
        header = pack_frame_length(payload_len)
        self.assertEqual(len(header), HEADER_SIZE)
        self.assertEqual(unpack_frame_length(header), payload_len)

    def test_detection_result_encoding(self):
        result = VisionResult(
            frame_id=7,
            bytes_received=2048,
            image_width=640,
            image_height=480,
            model="fake-model.pt",
            detections=(
                Detection(
                    class_id=0,
                    label="person",
                    confidence=0.91,
                    bbox_xyxy=(120.5, 44.0, 301.2, 420.7),
                    mask_polygon_xy=((121.0, 45.0), (300.0, 45.0), (301.0, 419.0)),
                ),
            ),
            latency_ms=18.4,
        )

        payload = json.loads(result.to_json_line())

        self.assertEqual(payload["frame_id"], 7)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["bytes_received"], 2048)
        self.assertEqual(payload["image_width"], 640)
        self.assertEqual(payload["image_height"], 480)
        self.assertEqual(payload["model"], "fake-model.pt")
        self.assertEqual(payload["latency_ms"], 18.4)
        self.assertEqual(
            payload["detections"],
            [
                {
                    "class_id": 0,
                    "label": "person",
                    "confidence": 0.91,
                    "bbox_xyxy": [120.5, 44.0, 301.2, 420.7],
                    "mask_polygon_xy": [[121.0, 45.0], [300.0, 45.0], [301.0, 419.0]],
                }
            ],
        )

    def test_empty_detections_are_serialized(self):
        result = VisionResult(
            frame_id=1,
            image_width=640,
            image_height=480,
            model="fake-model.pt",
            detections=(),
        )

        payload = json.loads(result.to_json_line())

        self.assertEqual(payload["detections"], [])

    def test_error_result_encoding(self):
        result = VisionResult.error_result(
            frame_id=3,
            bytes_received=128,
            error="invalid_image",
            message="Frame payload is not a decodable image",
        )

        payload = json.loads(result.to_json_line())

        self.assertEqual(payload["frame_id"], 3)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["bytes_received"], 128)
        self.assertEqual(payload["error"], "invalid_image")
        self.assertEqual(payload["message"], "Frame payload is not a decodable image")

    def test_detection_requires_four_bbox_coordinates(self):
        with self.assertRaises(ValueError):
            Detection(
                class_id=1,
                label="bad",
                confidence=0.5,
                bbox_xyxy=(1.0, 2.0, 3.0),
            )


if __name__ == "__main__":
    unittest.main()
