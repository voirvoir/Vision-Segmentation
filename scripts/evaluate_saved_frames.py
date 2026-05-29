from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from PIL import Image

from vision_backend.detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DETECTIONS,
    DetectionResult,
    UltralyticsDetector,
)
from vision_backend.protocol import Detection, VisionResult

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation")
SAVED_FRAME_CSV_FIELDS = [
    "frame_index",
    "image",
    "image_path",
    "image_bytes",
    "image_width",
    "image_height",
    "model",
    "device",
    "latency_ms",
    "detections",
    "response_bytes",
    "mask_point_count",
]


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def apply_limit(images: list[Path], limit: int | None) -> list[Path]:
    if limit is None or limit <= 0:
        return images
    return images[:limit]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def count_mask_points(detections: Iterable[Detection]) -> int:
    return sum(
        len(detection.mask_polygon_xy)
        for detection in detections
        if detection.mask_polygon_xy is not None
    )


def response_size_bytes(
    frame_index: int,
    image_bytes: int,
    detection_result: DetectionResult,
) -> int:
    response = VisionResult(
        frame_id=frame_index,
        bytes_received=image_bytes,
        image_width=detection_result.image_width,
        image_height=detection_result.image_height,
        model=detection_result.model,
        detections=detection_result.detections,
        latency_ms=round(detection_result.latency_ms, 2),
    )
    return len(response.to_json_line())


def build_saved_frame_record(
    frame_index: int,
    image_path: Path,
    image_bytes: int,
    device: str,
    detection_result: DetectionResult,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "image": image_path.name,
        "image_path": str(image_path),
        "image_bytes": image_bytes,
        "image_width": detection_result.image_width,
        "image_height": detection_result.image_height,
        "model": detection_result.model,
        "device": device,
        "latency_ms": round(detection_result.latency_ms, 3),
        "detections": len(detection_result.detections),
        "response_bytes": response_size_bytes(
            frame_index,
            image_bytes,
            detection_result,
        ),
        "mask_point_count": count_mask_points(detection_result.detections),
    }


def _numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _stats(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = _numeric_values(records, key)
    if not values:
        return {
            f"{key}_mean": None,
            f"{key}_median": None,
            f"{key}_min": None,
            f"{key}_max": None,
        }

    return {
        f"{key}_mean": round(mean(values), 3),
        f"{key}_median": round(median(values), 3),
        f"{key}_min": round(min(values), 3),
        f"{key}_max": round(max(values), 3),
    }


def summarize_saved_frame_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "frame_count": len(records),
        "total_image_bytes": int(sum(_numeric_values(records, "image_bytes"))),
        "total_response_bytes": int(sum(_numeric_values(records, "response_bytes"))),
        "total_detections": int(sum(_numeric_values(records, "detections"))),
        "total_mask_points": int(sum(_numeric_values(records, "mask_point_count"))),
    }
    for key in ("latency_ms", "image_bytes", "response_bytes", "detections"):
        summary.update(_stats(records, key))
    return summary


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_json(
    path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": summary,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_saved_frames(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    images = apply_limit(iter_images(args.frames), args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.frames}")

    detector = UltralyticsDetector(
        model_name=args.model,
        device=args.device,
        conf_threshold=args.conf_threshold,
        max_detections=args.max_detections,
        image_size=args.imgsz,
    )

    records: list[dict[str, Any]] = []
    for frame_index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            detection_result = detector.detect(image.convert("RGB"))

        record = build_saved_frame_record(
            frame_index=frame_index,
            image_path=image_path,
            image_bytes=image_path.stat().st_size,
            device=detector.device,
            detection_result=detection_result,
        )
        records.append(record)
        print(
            f"{record['frame_index']:04d} | {record['image']} | "
            f"{record['latency_ms']:.2f} ms | {record['detections']} detections | "
            f"{record['response_bytes']} response bytes"
        )

    summary = summarize_saved_frame_records(records)
    output_stem = f"saved_frames_{utc_timestamp()}"
    csv_path = args.output_dir / f"{output_stem}.csv"
    json_path = args.output_dir / f"{output_stem}.json"

    config = {
        "frames": str(args.frames),
        "model": args.model,
        "device": detector.device,
        "conf_threshold": args.conf_threshold,
        "max_detections": args.max_detections,
        "imgsz": args.imgsz,
        "limit": args.limit,
    }
    write_csv(csv_path, records, SAVED_FRAME_CSV_FIELDS)
    write_json(json_path, config, summary, records)

    print()
    print(f"Frames: {summary['frame_count']}")
    print(f"Latency mean: {summary['latency_ms_mean']} ms")
    print(f"Latency median: {summary['latency_ms_median']} ms")
    print(f"Response bytes mean: {summary['response_bytes_mean']}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return csv_path, json_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detector latency and response size on saved frames"
    )
    parser.add_argument("--frames", default=Path("logs/frames"), type=Path)
    parser.add_argument("--model", default="yolo26s-seg.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-threshold", default=DEFAULT_CONF_THRESHOLD, type=float)
    parser.add_argument("--max-detections", default=DEFAULT_MAX_DETECTIONS, type=int)
    parser.add_argument("--imgsz", default=DEFAULT_IMAGE_SIZE, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser.parse_args()


def main() -> None:
    evaluate_saved_frames(parse_args())


if __name__ == "__main__":
    main()
