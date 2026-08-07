"""Minimal Linux/UVC reader for the YDLIDAR (ASCamera) HP60C depth stream."""

from __future__ import annotations

import glob
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


DEPTH_FOURCCS = ("Y16 ", "Z16 ")


def _video_index(path: str) -> int:
    name = Path(path).name
    if not name.startswith("video") or not name[5:].isdigit():
        raise ValueError(f"A /dev/videoN path is required, got: {path}")
    return int(name[5:])


def _formats_for(device: str) -> str:
    """Return v4l2 format information when v4l2-ctl is available."""
    if shutil.which("v4l2-ctl") is None:
        return ""
    result = subprocess.run(
        ["v4l2-ctl", "--device", device, "--list-formats-ext"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def discover_depth_devices() -> list[str]:
    """Find V4L2 nodes which advertise a 16-bit depth/greyscale stream."""
    devices = [
        device
        for device in glob.glob("/dev/video*")
        if Path(device).name[5:].isdigit()
    ]
    devices.sort(key=_video_index)
    matches = []
    for device in devices:
        formats = _formats_for(device)
        if any(fourcc.strip() in formats for fourcc in DEPTH_FOURCCS):
            matches.append(device)
    return matches


def decode_depth_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Normalize OpenCV's possible RAW16 layouts to a uint16 HxW depth map."""
    if frame is None:
        raise ValueError("Empty depth frame")

    array = np.asarray(frame)
    if array.dtype == np.uint16:
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 2:
            raise ValueError(f"Unexpected uint16 frame shape: {array.shape}")
        return np.ascontiguousarray(array)

    if array.dtype == np.uint8:
        # Depending on the OpenCV/V4L2 build, Y16 can arrive as two bytes per
        # pixel in either HxWx2 or Hx(2W) form. UVC uses little-endian samples.
        if array.ndim == 3 and array.shape[2] == 2:
            packed = np.ascontiguousarray(array)
            return packed.view("<u2").reshape(array.shape[:2])
        if array.ndim == 2 and array.shape[1] == width * 2:
            packed = np.ascontiguousarray(array)
            return packed.view("<u2").reshape(array.shape[0], width)

    raise ValueError(
        f"Expected RAW16/Y16 data, got dtype={array.dtype}, shape={array.shape}. "
        "Check that the HP60C depth node (not its RGB node) was selected."
    )


def depth_at(
    depth: np.ndarray,
    x: int,
    y: int,
    *,
    radius: int = 2,
    scale: float = 0.001,
    min_m: float = 0.2,
    max_m: float = 4.0,
) -> float | None:
    """Return robust metric depth around a pixel, ignoring invalid samples."""
    if depth.ndim != 2:
        raise ValueError("depth must be a 2-D array")
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        return None

    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    values_m = depth[y0:y1, x0:x1].astype(np.float32) * scale
    valid = values_m[(values_m >= min_m) & (values_m <= max_m)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


class HP60CCamera:
    """Read metric depth maps from an HP60C V4L2 depth node."""

    def __init__(
        self,
        device: str | int | None = None,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 20,
    ) -> None:
        if device is None:
            detected = discover_depth_devices()
            if not detected:
                raise RuntimeError(
                    "HP60C RAW16 device not found. Connect the camera and run "
                    "'v4l2-ctl --list-devices' to identify its depth /dev/videoN node."
                )
            device = detected[0]

        self.device = f"/dev/video{device}" if isinstance(device, int) else device
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> "HP60CCamera":
        capture = cv2.VideoCapture(_video_index(self.device), cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open HP60C depth device: {self.device}")

        capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        return self

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read a depth frame from {self.device}")
        return decode_depth_frame(frame, self.width)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "HP60CCamera":
        return self.open()

    def __exit__(self, *_args: object) -> None:
        self.close()
