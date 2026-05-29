from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from .protocol import Detection

DEFAULT_MODEL = "yolo26s-seg.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_MAX_DETECTIONS = 50
DEFAULT_IMAGE_SIZE = 640


@dataclass(frozen=True)
class DetectionResult:
    image_width: int
    image_height: int
    model: str
    detections: tuple[Detection, ...]
    latency_ms: float


class ObjectDetector(Protocol):
    model_name: str

    def detect(self, image: Image.Image) -> DetectionResult:
        ...


def decode_image(frame_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(frame_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Frame payload is not a decodable image") from exc


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class UltralyticsDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install detection dependencies with "
                "`uv sync --extra detection`, then install a CUDA-enabled PyTorch "
                "build if you want GPU inference."
            ) from exc

        self.model_name = model_name
        self.device = resolve_device(device)
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.image_size = image_size
        self._model = YOLO(model_name)

    def detect(self, image: Image.Image) -> DetectionResult:
        rgb_image = image.convert("RGB")
        started = perf_counter()
        results = self._model.predict(
            rgb_image,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )
        latency_ms = (perf_counter() - started) * 1000.0

        detections = _detections_from_ultralytics_result(results[0])
        width, height = rgb_image.size
        return DetectionResult(
            image_width=width,
            image_height=height,
            model=self.model_name,
            detections=detections,
            latency_ms=latency_ms,
        )


def _detections_from_ultralytics_result(result: object) -> tuple[Detection, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()

    xyxy = getattr(boxes, "xyxy", None)
    confidences = getattr(boxes, "conf", None)
    classes = getattr(boxes, "cls", None)
    if xyxy is None or confidences is None or classes is None:
        return ()

    names = getattr(result, "names", {}) or {}
    xyxy_values = xyxy.cpu().tolist()
    confidence_values = confidences.cpu().tolist()
    class_values = classes.cpu().tolist()
    mask_polygons = _mask_polygons_from_ultralytics_result(result)

    detections: list[Detection] = []
    for index, (coords, confidence, class_value) in enumerate(
        zip(xyxy_values, confidence_values, class_values)
    ):
        class_id = int(class_value)
        label = str(names.get(class_id, class_id))
        detections.append(
            Detection(
                class_id=class_id,
                label=label,
                confidence=round(float(confidence), 4),
                bbox_xyxy=tuple(round(float(value), 2) for value in coords),
                mask_polygon_xy=mask_polygons[index] if index < len(mask_polygons) else None,
            )
        )

    return tuple(detections)


def _mask_polygons_from_ultralytics_result(
    result: object,
) -> list[tuple[tuple[float, float], ...]]:
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xy", None)
    if not polygons:
        return []

    mask_polygons: list[tuple[tuple[float, float], ...]] = []
    for polygon in polygons:
        points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
        mask_polygons.append(
            tuple(
                (round(float(x), 2), round(float(y), 2))
                for x, y in points
            )
        )

    return mask_polygons
