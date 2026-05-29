from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

from PIL import Image

from vision_backend.detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DETECTIONS,
    UltralyticsDetector,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def benchmark_model(args: argparse.Namespace, model_name: str, images: list[Path]) -> None:
    detector = UltralyticsDetector(
        model_name=model_name,
        device=args.device,
        conf_threshold=args.conf_threshold,
        max_detections=args.max_detections,
        image_size=args.imgsz,
    )

    latencies: list[float] = []
    detection_counts: list[int] = []
    for image_path in images:
        with Image.open(image_path) as image:
            result = detector.detect(image.convert("RGB"))
        latencies.append(result.latency_ms)
        detection_counts.append(len(result.detections))
        print(
            f"{model_name} | {image_path.name} | "
            f"{result.latency_ms:.2f} ms | {len(result.detections)} detections"
        )

    print()
    print(f"Model: {model_name}")
    print(f"Frames: {len(images)}")
    print(f"Latency mean: {mean(latencies):.2f} ms")
    print(f"Latency median: {median(latencies):.2f} ms")
    print(f"Detections mean: {mean(detection_counts):.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO segmentation on saved frames")
    parser.add_argument("--frames", default=Path("logs/frames"), type=Path)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model to benchmark; repeat to compare multiple models",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-threshold", default=DEFAULT_CONF_THRESHOLD, type=float)
    parser.add_argument("--max-detections", default=DEFAULT_MAX_DETECTIONS, type=int)
    parser.add_argument("--imgsz", default=DEFAULT_IMAGE_SIZE, type=int)
    args = parser.parse_args()
    args.model = args.model or ["yolo26s-seg.pt", "yolo26n-seg.pt"]

    images = iter_images(args.frames)
    if not images:
        raise SystemExit(f"No images found in {args.frames}")

    for model_name in args.model:
        benchmark_model(args, model_name, images)


if __name__ == "__main__":
    main()
