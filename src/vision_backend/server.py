from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import suppress

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
                    f"Max allowed is {max_frame_bytes}.\n"
                )
                writer.write(message.encode("utf-8"))
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

            result = VisionResult(frame_id=frame_id, bytes_received=len(frame))
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
) -> None:
    server = await asyncio.start_server(
        lambda r, w: handle_client(
            r,
            w,
            max_frame_bytes=max_frame_bytes,
            frame_log_dir=frame_log_dir,
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
        asyncio.run(
            run_server(
                host=args.host,
                port=args.port,
                max_frame_bytes=args.max_frame_bytes,
                frame_log_dir=args.frame_log_dir,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("Server stopped by user")


if __name__ == "__main__":
    main()
