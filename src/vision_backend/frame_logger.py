from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import re

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class SavedFrame:
    path: Path
    was_jpeg: bool


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", value)


def save_frame_for_debug(
    frame_bytes: bytes,
    frame_log_dir: str | Path,
    frame_id: int,
    peer: object,
) -> SavedFrame:
    output_dir = Path(frame_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peer_text = _sanitize_for_filename(str(peer))
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"frame_{stamp}_{peer_text}_{frame_id:06d}"

    try:
        with Image.open(BytesIO(frame_bytes)) as image:
            rgb_image = image.convert("RGB")
            output_path = output_dir / f"{base_name}.jpg"
            rgb_image.save(output_path, format="JPEG", quality=90)
            return SavedFrame(path=output_path, was_jpeg=True)
    except (UnidentifiedImageError, OSError, ValueError):
        output_path = output_dir / f"{base_name}.bin"
        output_path.write_bytes(frame_bytes)
        return SavedFrame(path=output_path, was_jpeg=False)