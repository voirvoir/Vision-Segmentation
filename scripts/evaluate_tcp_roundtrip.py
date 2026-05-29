from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from vision_backend.protocol import pack_frame_length

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation")
TCP_CSV_FIELDS = [
    "frame_index",
    "image",
    "image_path",
    "request_bytes",
    "response_bytes",
    "status",
    "detections",
    "backend_latency_ms",
    "client_round_trip_ms",
    "error",
]


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_frame_paths(images: list[Path], count: int | None) -> list[Path]:
    if not images:
        return []

    if count is None or count <= 0:
        return images

    return [images[index % len(images)] for index in range(count)]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def connect_socket(host: str, port: int, timeout_seconds: float) -> socket.socket:
    connection = socket.create_connection((host, port), timeout=timeout_seconds)
    connection.settimeout(timeout_seconds)
    return connection


def read_response_line(connection: socket.socket, max_response_bytes: int) -> bytes:
    chunks: list[bytes] = []
    bytes_read = 0

    while bytes_read < max_response_bytes:
        chunk = connection.recv(min(4096, max_response_bytes - bytes_read))
        if not chunk:
            break

        newline_index = chunk.find(b"\n")
        if newline_index >= 0:
            chunks.append(chunk[:newline_index])
            return b"".join(chunks).rstrip(b"\r")

        chunks.append(chunk)
        bytes_read += len(chunk)

    if bytes_read >= max_response_bytes:
        raise OSError(f"Backend response exceeded {max_response_bytes} bytes")

    if not chunks:
        raise OSError("Backend closed connection before sending a response")

    return b"".join(chunks).rstrip(b"\r")


def parse_response_payload(response_line: bytes) -> dict[str, Any]:
    try:
        return json.loads(response_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "error": "invalid_response",
            "message": str(exc),
            "detections": [],
        }


def detection_count(payload: dict[str, Any]) -> int:
    detections = payload.get("detections")
    return len(detections) if isinstance(detections, list) else 0


def send_frame(
    connection: socket.socket,
    frame_index: int,
    image_path: Path,
    max_response_bytes: int,
) -> dict[str, Any]:
    frame_bytes = image_path.read_bytes()
    request_bytes = len(pack_frame_length(len(frame_bytes))) + len(frame_bytes)

    started = time.perf_counter()
    connection.sendall(pack_frame_length(len(frame_bytes)) + frame_bytes)
    response_line = read_response_line(connection, max_response_bytes)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    payload = parse_response_payload(response_line)
    return {
        "frame_index": frame_index,
        "image": image_path.name,
        "image_path": str(image_path),
        "request_bytes": request_bytes,
        "response_bytes": len(response_line),
        "status": payload.get("status", ""),
        "detections": detection_count(payload),
        "backend_latency_ms": payload.get("latency_ms"),
        "client_round_trip_ms": round(elapsed_ms, 3),
        "error": payload.get("error", ""),
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


def summarize_tcp_records(
    records: list[dict[str, Any]],
    reconnect_each_frame: bool,
) -> dict[str, Any]:
    ok_count = sum(1 for record in records if record.get("status") == "ok")
    summary: dict[str, Any] = {
        "frame_count": len(records),
        "ok_count": ok_count,
        "error_count": len(records) - ok_count,
        "reconnect_each_frame": reconnect_each_frame,
        "total_request_bytes": int(sum(_numeric_values(records, "request_bytes"))),
        "total_response_bytes": int(sum(_numeric_values(records, "response_bytes"))),
        "total_detections": int(sum(_numeric_values(records, "detections"))),
    }
    for key in (
        "client_round_trip_ms",
        "backend_latency_ms",
        "request_bytes",
        "response_bytes",
        "detections",
    ):
        summary.update(_stats(records, key))
    return summary


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=TCP_CSV_FIELDS, extrasaction="ignore")
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


def evaluate_tcp_roundtrip(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    images = iter_images(args.frames)
    frame_paths = select_frame_paths(images, args.count)
    if not frame_paths:
        raise SystemExit(f"No images found in {args.frames}")

    records: list[dict[str, Any]] = []
    connection: socket.socket | None = None
    try:
        for frame_index, image_path in enumerate(frame_paths, start=1):
            if args.reconnect_each_frame or connection is None:
                if connection is not None:
                    connection.close()
                connection = connect_socket(args.host, args.port, args.timeout)

            try:
                record = send_frame(
                    connection=connection,
                    frame_index=frame_index,
                    image_path=image_path,
                    max_response_bytes=args.max_response_bytes,
                )
            except Exception as exc:
                record = {
                    "frame_index": frame_index,
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "request_bytes": image_path.stat().st_size + 4,
                    "response_bytes": 0,
                    "status": "error",
                    "detections": 0,
                    "backend_latency_ms": None,
                    "client_round_trip_ms": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if connection is not None:
                    connection.close()
                    connection = None

            records.append(record)
            print(
                f"{record['frame_index']:04d} | {record['image']} | "
                f"{record['status']} | rtt={record['client_round_trip_ms']} ms | "
                f"backend={record['backend_latency_ms']} ms | det={record['detections']}"
            )

            if args.delay > 0 and frame_index < len(frame_paths):
                time.sleep(args.delay)
    finally:
        if connection is not None:
            connection.close()

    summary = summarize_tcp_records(records, args.reconnect_each_frame)
    output_stem = f"tcp_roundtrip_{utc_timestamp()}"
    csv_path = args.output_dir / f"{output_stem}.csv"
    json_path = args.output_dir / f"{output_stem}.json"

    config = {
        "host": args.host,
        "port": args.port,
        "frames": str(args.frames),
        "count": args.count,
        "delay": args.delay,
        "timeout": args.timeout,
        "max_response_bytes": args.max_response_bytes,
        "reconnect_each_frame": args.reconnect_each_frame,
    }
    write_csv(csv_path, records)
    write_json(json_path, config, summary, records)

    print()
    print(f"Frames: {summary['frame_count']}")
    print(f"OK: {summary['ok_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Round trip mean: {summary['client_round_trip_ms_mean']} ms")
    print(f"Backend latency mean: {summary['backend_latency_ms_mean']} ms")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return csv_path, json_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TCP round-trip latency against a running vision backend"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=40081, type=int)
    parser.add_argument("--frames", default=Path("logs/frames"), type=Path)
    parser.add_argument("--count", default=None, type=int)
    parser.add_argument("--delay", default=0.0, type=float)
    parser.add_argument("--timeout", default=10.0, type=float)
    parser.add_argument("--max-response-bytes", default=262144, type=int)
    parser.add_argument("--reconnect-each-frame", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser.parse_args()


def main() -> None:
    evaluate_tcp_roundtrip(parse_args())


if __name__ == "__main__":
    main()
