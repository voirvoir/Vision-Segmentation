from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from vision_backend.frame_logger import save_frame_for_debug


class FrameLoggerTests(unittest.TestCase):
    def test_saves_decodable_image_as_jpeg(self):
        with TemporaryDirectory() as tmp_dir:
            image_buffer = BytesIO()
            image = Image.new("RGB", (8, 8), color=(255, 0, 0))
            image.save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()

            saved = save_frame_for_debug(
                frame_bytes=image_bytes,
                frame_log_dir=tmp_dir,
                frame_id=1,
                peer=("127.0.0.1", 40081),
            )

            self.assertTrue(saved.was_jpeg)
            self.assertEqual(saved.path.suffix, ".jpg")
            self.assertTrue(saved.path.exists())

            with Image.open(saved.path) as output_image:
                self.assertEqual(output_image.format, "JPEG")

    def test_saves_raw_bin_when_not_decodable_image(self):
        with TemporaryDirectory() as tmp_dir:
            data = b"not-an-image"

            saved = save_frame_for_debug(
                frame_bytes=data,
                frame_log_dir=tmp_dir,
                frame_id=2,
                peer=("127.0.0.1", 40081),
            )

            self.assertFalse(saved.was_jpeg)
            self.assertEqual(saved.path.suffix, ".bin")
            self.assertEqual(saved.path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()