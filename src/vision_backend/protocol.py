from __future__ import annotations

import json
import struct
from dataclasses import dataclass

HEADER_SIZE = 4


@dataclass(frozen=True)
class VisionResult:
    frame_id: int
    bytes_received: int
    status: str = "ok"

    def to_json_line(self) -> bytes:
        payload = {
            "frame_id": self.frame_id,
            "bytes_received": self.bytes_received,
            "status": self.status,
        }
        return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def pack_frame_length(length: int) -> bytes:
    if length < 0:
        raise ValueError("Frame length must be non-negative")
    return struct.pack("<I", length)


def unpack_frame_length(header: bytes) -> int:
    if len(header) != HEADER_SIZE:
        raise ValueError("Frame header must be exactly 4 bytes")
    return struct.unpack("<I", header)[0]
