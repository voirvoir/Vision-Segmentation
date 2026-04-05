import unittest

from vision_backend.protocol import HEADER_SIZE, VisionResult, pack_frame_length, unpack_frame_length


class ProtocolTests(unittest.TestCase):
    def test_length_header_roundtrip(self):
        payload_len = 123456
        header = pack_frame_length(payload_len)
        self.assertEqual(len(header), HEADER_SIZE)
        self.assertEqual(unpack_frame_length(header), payload_len)

    def test_vision_result_encoding(self):
        result = VisionResult(frame_id=7, bytes_received=2048)
        encoded = result.to_json_line().decode("utf-8")
        self.assertEqual(encoded, '{"frame_id":7,"bytes_received":2048,"status":"ok"}\n')


if __name__ == "__main__":
    unittest.main()
