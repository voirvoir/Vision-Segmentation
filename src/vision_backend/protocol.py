from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

HEADER_SIZE = 4


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_polygon_xy: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly four coordinates")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }
        if self.mask_polygon_xy is not None:
            payload["mask_polygon_xy"] = [
                [x, y] for x, y in self.mask_polygon_xy
            ]
        return payload


@dataclass(frozen=True)
class VisionResult:
    frame_id: int
    status: str = "ok"
    bytes_received: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    model: str | None = None
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    latency_ms: float | None = None
    error: str | None = None
    message: str | None = None

    @classmethod
    def error_result(
        cls,
        frame_id: int,
        error: str,
        message: str,
        bytes_received: int | None = None,
    ) -> "VisionResult":
        return cls(
            frame_id=frame_id,
            status="error",
            bytes_received=bytes_received,
            error=error,
            message=message,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_id": self.frame_id,
            "status": self.status,
        }

        if self.bytes_received is not None:
            payload["bytes_received"] = self.bytes_received

        if self.image_width is not None:
            payload["image_width"] = self.image_width

        if self.image_height is not None:
            payload["image_height"] = self.image_height

        if self.model is not None:
            payload["model"] = self.model

        if self.image_width is not None or self.image_height is not None or self.detections:
            payload["detections"] = [detection.to_payload() for detection in self.detections]

        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms

        if self.error is not None:
            payload["error"] = self.error

        if self.message is not None:
            payload["message"] = self.message

        return payload

    def to_json_line(self) -> bytes:
        payload = self.to_payload()
        return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def pack_frame_length(length: int) -> bytes:
    if length < 0:
        raise ValueError("Frame length must be non-negative")
    return struct.pack("<I", length)


def unpack_frame_length(header: bytes) -> int:
    if len(header) != HEADER_SIZE:
        raise ValueError("Frame header must be exactly 4 bytes")
    return struct.unpack("<I", header)[0]
