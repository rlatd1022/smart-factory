#!/usr/bin/env python3
"""Real-time RGB/depth viewer for the NOVATEK/ASJ HP60C camera.

The HP60C exposes a vendor-specific UVC MJPG stream. Each packet contains a
real RGB JPEG followed by a 640x480 uint16 depth map. The upper four bits of a
depth sample are status flags and the lower twelve bits are distance in mm.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


WIDTH = 640
HEIGHT = 480
UVC_HEIGHT = 642
DEPTH_BYTES = WIDTH * HEIGHT * 2
POST_JPEG_PADDING = 512
JPEG_SIGNATURE = b"\xff\xd8\xff\xdb\x00\x84\x00\x06\x04\x05"
WINDOW = "HP60C live - RGB | depth (click to measure, q to quit)"


def _video_number(path: str) -> int:
    return int(Path(path).name.removeprefix("video"))


def find_hp60c_device() -> str:
    """Return the ASJ video node which advertises the 640x642 MJPG stream."""
    devices = [
        path
        for path in glob.glob("/dev/video*")
        if Path(path).name.removeprefix("video").isdigit()
    ]
    for device in sorted(devices, key=_video_number):
        info = subprocess.run(
            ["v4l2-ctl", "--device", device, "--info"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if "ASJ ZNX_NVT" not in info:
            continue
        formats = subprocess.run(
            ["v4l2-ctl", "--device", device, "--list-formats-ext"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if "MJPG" in formats and "640x642" in formats:
            return device
    raise RuntimeError(
        "HP60C video stream not found. Reconnect the ASJ ZNX_NVT camera and "
        "check 'v4l2-ctl --list-devices'."
    )


def decode_packet(packet: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode one HP60C vendor packet into BGR and depth-mm arrays."""
    jpeg_start = packet.find(JPEG_SIGNATURE)
    if jpeg_start < 0:
        raise ValueError("RGB JPEG signature not found")
    jpeg_end = packet.find(b"\xff\xd9", jpeg_start + 2)
    if jpeg_end < 0:
        raise ValueError("RGB JPEG is incomplete")

    depth_start = jpeg_end + 2 + POST_JPEG_PADDING
    depth_end = depth_start + DEPTH_BYTES
    if len(packet) < depth_end:
        raise ValueError("Depth payload is incomplete")

    jpeg_data = np.frombuffer(packet[jpeg_start : jpeg_end + 2], np.uint8)
    rgb = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError("RGB JPEG could not be decoded as 640x480")

    raw_depth = np.frombuffer(
        packet[depth_start:depth_end], dtype="<u2"
    ).reshape(HEIGHT, WIDTH)
    depth_mm = np.bitwise_and(raw_depth, 0x0FFF).astype(np.uint16)
    return rgb, depth_mm


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    sequence: int


class HP60CStream:
    """Continuously read and decode the HP60C stream using v4l2-ctl mmap."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._latest: Frame | None = None
        self._closed = False
        self.error: str | None = None

    def start(self) -> "HP60CStream":
        self._process = subprocess.Popen(
            [
                "v4l2-ctl",
                "--device",
                self.device,
                f"--set-fmt-video=width={WIDTH},height={UVC_HEIGHT},pixelformat=MJPG",
                "--stream-mmap=3",
                "--stream-to=-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        data = bytearray()
        sequence = 0
        try:
            while not self._closed:
                chunk = os.read(self._process.stdout.fileno(), 262144)
                if not chunk:
                    break
                data.extend(chunk)

                while True:
                    jpeg_start = data.find(JPEG_SIGNATURE)
                    if jpeg_start < 0:
                        if len(data) > 3_000_000:
                            del data[:-4096]
                        break
                    jpeg_end = data.find(b"\xff\xd9", jpeg_start + 2)
                    if jpeg_end < 0:
                        if jpeg_start > 2560:
                            del data[: jpeg_start - 2560]
                        break
                    packet_end = jpeg_end + 2 + POST_JPEG_PADDING + DEPTH_BYTES
                    if len(data) < packet_end:
                        break

                    packet = bytes(data[:packet_end])
                    del data[:packet_end]
                    try:
                        rgb, depth = decode_packet(packet)
                    except ValueError:
                        continue
                    sequence += 1
                    with self._condition:
                        self._latest = Frame(rgb, depth, sequence)
                        self._condition.notify_all()
        except OSError as exc:
            if not self._closed:
                self.error = str(exc)
        finally:
            if not self._closed and self.error is None:
                stderr = b""
                if self._process.stderr is not None:
                    stderr = self._process.stderr.read()
                self.error = stderr.decode(errors="replace").strip() or "Stream stopped"
            with self._condition:
                self._condition.notify_all()

    def read(self, after: int = 0, timeout: float = 1.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                not self._closed
                and self.error is None
                and (self._latest is None or self._latest.sequence <= after)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._latest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
        with self._condition:
            self._condition.notify_all()

    def __enter__(self) -> "HP60CStream":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def depth_at(depth: np.ndarray, x: int, y: int, radius: int = 3) -> float | None:
    """Median valid depth in millimetres near a pixel."""
    x0, x1 = max(0, x - radius), min(WIDTH, x + radius + 1)
    y0, y1 = max(0, y - radius), min(HEIGHT, y + radius + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[(patch >= 200) & (patch <= 4000)]
    return None if valid.size == 0 else float(np.median(valid))


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = (depth >= 200) & (depth <= 4000)
    scaled = np.zeros(depth.shape, np.uint8)
    scaled[valid] = np.clip(
        (4000.0 - depth[valid]) * 255.0 / 3800.0, 0, 255
    ).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Live HP60C RGB/depth viewer")
    parser.add_argument("--device", help="HP60C node; auto-detected by default")
    parser.add_argument(
        "--check", action="store_true", help="Read one frame, print stats, and exit"
    )
    args = parser.parse_args()

    device = args.device or find_hp60c_device()
    print(f"HP60C device: {device}")
    selected = [WIDTH // 2, HEIGHT // 2]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selected[0] = x % WIDTH
        selected[1] = min(y, HEIGHT - 1)

    if not args.check:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        with HP60CStream(device) as stream:
            sequence = 0
            while True:
                frame = stream.read(sequence, timeout=0.1)
                if frame is None or frame.sequence <= sequence:
                    if stream.error:
                        raise RuntimeError(stream.error)
                    if not args.check and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue
                sequence = frame.sequence

                if args.check:
                    valid = frame.depth[
                        (frame.depth >= 200) & (frame.depth <= 4000)
                    ]
                    print(
                        f"RGB={frame.rgb.shape}, depth={frame.depth.shape}, "
                        f"valid={valid.size}/{frame.depth.size}, "
                        f"median={float(np.median(valid)) if valid.size else 'N/A'} mm"
                    )
                    return 0

                depth_view = colorize_depth(frame.depth)
                x, y = selected
                distance = depth_at(frame.depth, x, y)
                label = "N/A" if distance is None else f"{distance:.0f} mm"
                for image in (frame.rgb, depth_view):
                    cv2.drawMarker(
                        image, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2
                    )
                    cv2.putText(
                        image,
                        f"({x}, {y}) {label}",
                        (max(8, x - 80), max(25, y - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame.rgb,
                    "RGB",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    depth_view,
                    "DEPTH (200-4000 mm)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(WINDOW, np.hstack((frame.rgb, depth_view)))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
