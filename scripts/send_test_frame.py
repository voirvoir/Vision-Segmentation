from __future__ import annotations

import argparse
import socket
import struct


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one test frame to vision server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=40081, type=int)
    parser.add_argument("--size", default=1024, type=int, help="Frame payload size in bytes")
    args = parser.parse_args()

    payload = bytes([42]) * args.size

    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        sock.sendall(struct.pack("<I", len(payload)))
        sock.sendall(payload)

        response = sock.recv(4096)
        print(response.decode("utf-8", errors="replace").strip())


if __name__ == "__main__":
    main()
