from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vision_backend.detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MODEL,
    DetectionResult,
    UltralyticsDetector,
)
from vision_backend.protocol import Detection, VisionResult

DEFAULT_OUTPUT_DIR = Path("outputs/detection_tests")


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (230, 57, 70),
        (29, 117, 209),
        (42, 157, 143),
        (245, 166, 35),
        (123, 97, 255),
        (0, 168, 150),
        (255, 111, 97),
        (80, 200, 120),
    )
    return palette[class_id % len(palette)]


def clamp_box(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0.0, min(float(image_width - 1), x1))
    y1 = max(0.0, min(float(image_height - 1), y1))
    x2 = max(0.0, min(float(image_width - 1), x2))
    y2 = max(0.0, min(float(image_height - 1), y2))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def draw_detections(image: Image.Image, detections: tuple[Detection, ...]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    font = ImageFont.load_default()
    width, height = annotated.size
    line_width = max(2, round(min(width, height) * 0.004))

    mask_overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_overlay)
    for detection in detections:
        if detection.mask_polygon_xy:
            color = color_for_class(detection.class_id)
            mask_draw.polygon(detection.mask_polygon_xy, fill=color + (80,))

    annotated = Image.alpha_composite(annotated.convert("RGBA"), mask_overlay).convert("RGB")
    draw = ImageDraw.Draw(annotated)

    for detection in detections:
        box = clamp_box(detection.bbox_xyxy, width, height)
        color = color_for_class(detection.class_id)
        label = f"{detection.label} {detection.confidence:.2f}"

        if detection.mask_polygon_xy:
            draw.line(detection.mask_polygon_xy + (detection.mask_polygon_xy[0],), fill=color, width=line_width)

        draw.rectangle(box, outline=color, width=line_width)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = int(box[0])
        label_y = max(0, int(box[1]) - text_height - 6)
        label_rect = (
            label_x,
            label_y,
            label_x + text_width + 8,
            label_y + text_height + 6,
        )
        draw.rectangle(label_rect, fill=color)
        draw.text((label_x + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

    return annotated


def result_to_vision_response(
    frame_id: int,
    bytes_received: int,
    detection_result: DetectionResult,
) -> VisionResult:
    return VisionResult(
        frame_id=frame_id,
        bytes_received=bytes_received,
        image_width=detection_result.image_width,
        image_height=detection_result.image_height,
        model=detection_result.model,
        detections=detection_result.detections,
        latency_ms=round(detection_result.latency_ms, 2),
    )


def output_paths(
    image_path: Path,
    out_dir: Path,
    json_out: Path | None,
    image_out: Path | None,
) -> tuple[Path, Path]:
    stem = image_path.stem
    json_path = json_out or out_dir / f"{stem}.detections.json"
    annotated_path = image_out or out_dir / f"{stem}.annotated.jpg"
    return json_path, annotated_path


def write_json_response(path: Path, response: VisionResult, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        path.write_text(json.dumps(response.to_payload(), indent=2) + "\n", encoding="utf-8")
    else:
        path.write_bytes(response.to_json_line())


def run_detection(args: argparse.Namespace) -> tuple[Path, Path, VisionResult]:
    image_path = args.image
    image_bytes = image_path.read_bytes()

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    detector = UltralyticsDetector(
        model_name=args.model,
        device=args.device,
        conf_threshold=args.conf_threshold,
        max_detections=args.max_detections,
        image_size=args.imgsz,
    )
    detection_result = detector.detect(rgb_image)
    response = result_to_vision_response(
        frame_id=args.frame_id,
        bytes_received=len(image_bytes),
        detection_result=detection_result,
    )

    json_path, annotated_path = output_paths(
        image_path=image_path,
        out_dir=args.out_dir,
        json_out=args.json_out,
        image_out=args.image_out,
    )

    annotated_image = draw_detections(rgb_image, detection_result.detections)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_image.save(annotated_path, quality=95)
    write_json_response(json_path, response, pretty=args.pretty_json)

    return json_path, annotated_path, response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run instance segmentation on one image and write the Unity JSON response "
            "plus an annotated expected image."
        )
    )
    parser.add_argument("image", type=Path, help="Input JPEG/PNG image path")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--json-out", type=Path, help="Exact JSON output path")
    parser.add_argument("--image-out", type=Path, help="Exact annotated image output path")
    parser.add_argument("--frame-id", default=1, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-threshold", default=DEFAULT_CONF_THRESHOLD, type=float)
    parser.add_argument("--max-detections", default=DEFAULT_MAX_DETECTIONS, type=int)
    parser.add_argument("--imgsz", default=DEFAULT_IMAGE_SIZE, type=int)
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Write indented JSON instead of the exact compact server JSON line",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        json_path, annotated_path, response = run_detection(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"JSON: {json_path}")
    print(f"Annotated image: {annotated_path}")
    print(f"Detections: {len(response.detections)}")
    print(response.to_json_line().decode("utf-8").strip())


if __name__ == "__main__":
    main()
