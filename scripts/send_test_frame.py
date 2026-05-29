from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import socket
import struct

from PIL import Image


def make_generated_jpeg(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(40, 100, 180))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one test frame to vision server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=40081, type=int)
    parser.add_argument("--image", type=Path, help="Path to a JPEG/PNG frame to send")
    parser.add_argument("--width", default=640, type=int, help="Generated test image width")
    parser.add_argument("--height", default=480, type=int, help="Generated test image height")
    parser.add_argument(
        "--raw-size",
        type=int,
        help="Send non-image raw bytes of this size for invalid-image testing",
    )
    args = parser.parse_args()

    if args.raw_size is not None:
        payload = bytes([42]) * args.raw_size
    elif args.image is not None:
        payload = args.image.read_bytes()
    else:
        payload = make_generated_jpeg(args.width, args.height)

    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        sock.sendall(struct.pack("<I", len(payload)))
        sock.sendall(payload)

        response = sock.recv(4096)
        print(response.decode("utf-8", errors="replace").strip())


if __name__ == "__main__":
    main()
