from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from vision_backend.detector import DetectionResult
from vision_backend.protocol import Detection, pack_frame_length


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


saved_frames = load_script_module(
    "evaluate_saved_frames_script",
    "scripts/evaluate_saved_frames.py",
)
tcp_roundtrip = load_script_module(
    "evaluate_tcp_roundtrip_script",
    "scripts/evaluate_tcp_roundtrip.py",
)


class FakeSocket:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def recv(self, _: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


class EvaluationScriptTests(unittest.TestCase):
    def test_saved_frame_helpers_write_csv_and_json_summary(self):
        detection_result = DetectionResult(
            image_width=640,
            image_height=480,
            model="fake-model.pt",
            detections=(
                Detection(
                    class_id=63,
                    label="laptop",
                    confidence=0.51,
                    bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
                    mask_polygon_xy=((10.0, 20.0), (100.0, 20.0), (100.0, 200.0)),
                ),
            ),
            latency_ms=12.3456,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            Image.new("RGB", (640, 480), color=(20, 40, 80)).save(image_path)

            record = saved_frames.build_saved_frame_record(
                frame_index=1,
                image_path=image_path,
                image_bytes=image_path.stat().st_size,
                device="cuda:0",
                detection_result=detection_result,
            )
            summary = saved_frames.summarize_saved_frame_records([record])

            csv_path = Path(tmp_dir) / "out.csv"
            json_path = Path(tmp_dir) / "out.json"
            saved_frames.write_csv(csv_path, [record], saved_frames.SAVED_FRAME_CSV_FIELDS)
            saved_frames.write_json(json_path, {"model": "fake-model.pt"}, summary, [record])

            self.assertEqual(record["detections"], 1)
            self.assertEqual(record["mask_point_count"], 3)
            self.assertGreater(record["response_bytes"], 0)
            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["latency_ms_mean"], 12.346)
            self.assertTrue(csv_path.read_text(encoding="utf-8").startswith("frame_index,"))

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_detections"], 1)
            self.assertEqual(payload["records"][0]["model"], "fake-model.pt")

    def test_iter_images_filters_supported_extensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            Image.new("RGB", (4, 4)).save(root / "a.jpg")
            Image.new("RGB", (4, 4)).save(root / "b.png")
            (root / "notes.txt").write_text("not an image", encoding="utf-8")

            images = saved_frames.iter_images(root)
            self.assertEqual([path.name for path in images], ["a.jpg", "b.png"])

    def test_tcp_helpers_parse_response_line_and_summary(self):
        response = b'{"status":"ok","latency_ms":7.5,"detections":[{"label":"cup"}]}\n'
        fake_socket = FakeSocket([response[:12], response[12:]])

        line = tcp_roundtrip.read_response_line(fake_socket, 1024)
        payload = tcp_roundtrip.parse_response_payload(line)
        records = [
            {
                "status": payload["status"],
                "request_bytes": len(pack_frame_length(10)) + 10,
                "response_bytes": len(line),
                "detections": tcp_roundtrip.detection_count(payload),
                "backend_latency_ms": payload["latency_ms"],
                "client_round_trip_ms": 9.25,
            }
        ]
        summary = tcp_roundtrip.summarize_tcp_records(records, reconnect_each_frame=False)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(tcp_roundtrip.detection_count(payload), 1)
        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["client_round_trip_ms_mean"], 9.25)
        self.assertFalse(summary["reconnect_each_frame"])

    def test_select_frame_paths_cycles_to_requested_count(self):
        images = [Path("a.jpg"), Path("b.jpg")]

        selected = tcp_roundtrip.select_frame_paths(images, 5)

        self.assertEqual(
            [path.name for path in selected],
            ["a.jpg", "b.jpg", "a.jpg", "b.jpg", "a.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
