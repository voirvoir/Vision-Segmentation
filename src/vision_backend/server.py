from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import suppress

from .detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MODEL,
    ObjectDetector,
    UltralyticsDetector,
    decode_image,
)
from .frame_logger import save_frame_for_debug
from .protocol import VisionResult, unpack_frame_length

LOGGER = logging.getLogger("vision_backend.server")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 40081
DEFAULT_MAX_FRAME_BYTES = 10 * 1024 * 1024
DEFAULT_FRAME_LOG_DIR = "logs/frames"


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    max_frame_bytes: int,
    frame_log_dir: str,
    detector: ObjectDetector,
    inference_lock: asyncio.Lock,
    save_debug_frames: bool,
) -> None:
    peer = writer.get_extra_info("peername")
    LOGGER.info("Client connected: %s", peer)
    frame_id = 0

    try:
        while True:
            header = await reader.readexactly(4)
            frame_length = unpack_frame_length(header)

            if frame_length > max_frame_bytes:
                message = (
                    f"Frame too large ({frame_length} bytes). "
                    f"Max allowed is {max_frame_bytes}."
                )
                result = VisionResult.error_result(
                    frame_id=frame_id + 1,
                    error="frame_too_large",
                    message=message,
                )
                writer.write(result.to_json_line())
                await writer.drain()
                LOGGER.warning("Rejected oversized frame from %s: %d", peer, frame_length)
                break

            frame = await reader.readexactly(frame_length)
            frame_id += 1

            LOGGER.info(
                "Received frame %d from %s (%d bytes)",
                frame_id,
                peer,
                len(frame),
            )

            if save_debug_frames:
                saved_frame = await asyncio.to_thread(
                    save_frame_for_debug,
                    frame,
                    frame_log_dir,
                    frame_id,
                    peer,
                )
                if saved_frame.was_jpeg:
                    LOGGER.info("Saved frame %d as JPEG: %s", frame_id, saved_frame.path)
                else:
                    LOGGER.warning(
                        "Frame %d is not a decodable image; saved raw bytes: %s",
                        frame_id,
                        saved_frame.path,
                    )

            try:
                image = await asyncio.to_thread(decode_image, frame)
            except ValueError as exc:
                result = VisionResult.error_result(
                    frame_id=frame_id,
                    error="invalid_image",
                    message=str(exc),
                    bytes_received=len(frame),
                )
                writer.write(result.to_json_line())
                await writer.drain()
                continue

            try:
                async with inference_lock:
                    detection_result = await asyncio.to_thread(detector.detect, image)
            except Exception as exc:
                LOGGER.exception("Inference failed for frame %d from %s", frame_id, peer)
                result = VisionResult.error_result(
                    frame_id=frame_id,
                    error="inference_failed",
                    message=str(exc),
                    bytes_received=len(frame),
                )
                writer.write(result.to_json_line())
                await writer.drain()
                continue

            result = VisionResult(
                frame_id=frame_id,
                bytes_received=len(frame),
                image_width=detection_result.image_width,
                image_height=detection_result.image_height,
                model=detection_result.model,
                detections=detection_result.detections,
                latency_ms=round(detection_result.latency_ms, 2),
            )
            writer.write(result.to_json_line())
            await writer.drain()

    except asyncio.IncompleteReadError:
        LOGGER.info("Client disconnected: %s", peer)
    except ConnectionResetError:
        LOGGER.info("Client reset connection: %s", peer)
    except Exception:
        LOGGER.exception("Unexpected client error for %s", peer)
    finally:
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()


async def run_server(
    host: str,
    port: int,
    max_frame_bytes: int,
    frame_log_dir: str,
    detector: ObjectDetector,
    save_debug_frames: bool,
) -> None:
    inference_lock = asyncio.Lock()
    server = await asyncio.start_server(
        lambda r, w: handle_client(
            r,
            w,
            max_frame_bytes=max_frame_bytes,
            frame_log_dir=frame_log_dir,
            detector=detector,
            inference_lock=inference_lock,
            save_debug_frames=save_debug_frames,
        ),
        host,
        port,
    )

    socket_info = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    LOGGER.info("Vision TCP server listening on %s", socket_info)

    async with server:
        await server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision TCP backend server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="Bind port")
    parser.add_argument(
        "--max-frame-bytes",
        default=DEFAULT_MAX_FRAME_BYTES,
        type=int,
        help="Maximum accepted frame payload size",
    )
    parser.add_argument(
        "--frame-log-dir",
        default=DEFAULT_FRAME_LOG_DIR,
        help="Directory where received frames are dumped for debugging",
    )
    parser.add_argument(
        "--no-frame-log",
        action="store_true",
        help="Disable saving received frames to disk for lower live VR latency",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ultralytics model name or local checkpoint path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device, for example "auto", "cpu", or "cuda:0"',
    )
    parser.add_argument(
        "--conf-threshold",
        default=DEFAULT_CONF_THRESHOLD,
        type=float,
        help="Minimum detection confidence",
    )
    parser.add_argument(
        "--max-detections",
        default=DEFAULT_MAX_DETECTIONS,
        type=int,
        help="Maximum detections returned per frame",
    )
    parser.add_argument(
        "--imgsz",
        default=DEFAULT_IMAGE_SIZE,
        type=int,
        help="Detector inference image size",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        detector = UltralyticsDetector(
            model_name=args.model,
            device=args.device,
            conf_threshold=args.conf_threshold,
            max_detections=args.max_detections,
            image_size=args.imgsz,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc

    LOGGER.info(
        "Loaded detector model=%s device=%s conf=%.2f max_det=%d imgsz=%d",
        detector.model_name,
        detector.device,
        detector.conf_threshold,
        detector.max_detections,
        detector.image_size,
    )

    try:
        asyncio.run(
            run_server(
                host=args.host,
                port=args.port,
                max_frame_bytes=args.max_frame_bytes,
                frame_log_dir=args.frame_log_dir,
                detector=detector,
                save_debug_frames=not args.no_frame_log,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("Server stopped by user")


if __name__ == "__main__":
    main()
