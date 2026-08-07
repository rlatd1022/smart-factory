"""Depth Window Inspector for Smart Factory.

Integrates depth_calib HSV mask gating & eroded depth sampling at center point:
- Checks if a blue/sample object is present at the screen center (w//2, h//2)
- Samples depth only on eroded mask interior (avoids edges/conveyor belt)
- Tracks N/A -> DETECTED -> N/A transit sequence
- Finds minimum depth (highest point of sample)
- Evaluates:
    min_depth == 388 mm (±1 mm) -> Good part (OK, No Kick)
    min_depth != 388 mm (e.g. 390 mm) -> Defect (NG, Trigger Kicker 2)
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

SHM_PATH = "/dev/shm/hp60c_frames"
SHM_MAGIC = 0x48503630  # "HP60"
HEADER_SIZE = 64
MAX_RGB_SIZE = 1920 * 1080 * 3
MAX_DEPTH_SIZE = 640 * 480 * 2


@dataclass
class InspectionResult:
    event: str  # "IDLE", "MEASURING", "OBJECT_DONE"
    status: str  # "EMPTY", "MEASURE", "OK", "NG", "N/A"
    ok: Optional[bool]  # True=OK, False=NG, None=Measuring/Idle
    current_depth_mm: Optional[float]
    min_depth_mm: Optional[float]
    ref_depth_mm: float
    delta_mm: Optional[float]
    should_kick: bool
    kicker_num: int
    samples_count: int
    uv: Optional[Tuple[int, int]] = None
    mask_ok: bool = False


def color_mask(
    bgr: np.ndarray,
    lo: Tuple[int, int, int] = (102, 175, 118),
    hi: Tuple[int, int, int] = (179, 255, 255),
) -> np.ndarray:
    """HSV mask for blue/grey sample with morphology opening."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)


def mask_gate_stats(
    mask: np.ndarray, u: int, v: int, radius: int = 12, min_px: int = 20
) -> Tuple[bool, int]:
    """Check if mask pixel count inside circular gate around (u, v) is >= min_px."""
    h, w = mask.shape[:2]
    r = int(radius)
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    ys = np.arange(y0, y1, dtype=np.int32)[:, None]
    xs = np.arange(x0, x1, dtype=np.int32)[None, :]
    in_circle = (xs - u) ** 2 + (ys - v) ** 2 <= r * r
    patch = mask[y0:y1, x0:x1] > 0
    hits = int((patch & in_circle).sum())
    return hits >= int(min_px), hits


def mask_hits_near(
    mask: np.ndarray, u: int, v: int, radius: int = 12, min_px: int = 20
) -> bool:
    ok, _ = mask_gate_stats(mask, u, v, radius, min_px)
    return ok


def shrink_mask(mask: np.ndarray, erode_px: int = 5) -> np.ndarray:
    """Erode mask to keep only inner area, excluding unreliable boundary pixels."""
    if erode_px <= 0 or mask is None:
        return mask
    k = 2 * int(erode_px) + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kern)


def depth_at_uv(
    depth_map: np.ndarray,
    u: int,
    v: int,
    radius: int = 1,
    mask: Optional[np.ndarray] = None,
    min_mm: float = 200.0,
    max_mm: float = 4000.0,
) -> Optional[float]:
    """Sample median depth in mm within radius around (u, v), masked to sample interior."""
    if depth_map is None or depth_map.size == 0:
        return None

    h, w = depth_map.shape[:2]
    r = int(radius)
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    y0, y1 = max(0, v - r), min(h, v + r + 1)

    patch = depth_map[y0:y1, x0:x1].astype(np.float32)
    valid = (patch >= min_mm) & (patch <= max_mm)

    if mask is not None:
        m_patch = mask[y0:y1, x0:x1] > 0
        valid = valid & m_patch

    pts = patch[valid]
    if pts.size == 0:
        return None

    return float(np.median(pts))


class DepthWindowInspector:
    """Evaluates sample height by finding the minimum depth during object transit."""

    def __init__(
        self,
        ref_depth_mm: float = 388.0,
        tol_mm: float = 1.0,
        kicker_num: int = 2,
        min_samples: int = 3,
        idle_debounce_frames: int = 3,
        use_center: bool = True,
        uv: Optional[Tuple[int, int]] = (320, 240),
        sample_radius: int = 1,
        hsv_lo: Tuple[int, int, int] = (102, 175, 118),
        hsv_hi: Tuple[int, int, int] = (179, 255, 255),
        require_mask: bool = True,
        mask_radius_px: int = 12,
        min_mask_px: int = 20,
        depth_mask_erode_px: int = 5,
    ) -> None:
        self.ref_depth_mm = float(ref_depth_mm)
        self.tol_mm = float(tol_mm)
        self.kicker_num = int(kicker_num)
        self.min_samples = int(min_samples)
        self.idle_debounce_frames = int(idle_debounce_frames)
        self.use_center = bool(use_center)
        self.uv = uv
        self.sample_radius = int(sample_radius)
        self.hsv_lo = tuple(int(x) for x in hsv_lo)
        self.hsv_hi = tuple(int(x) for x in hsv_hi)
        self.require_mask = bool(require_mask)
        self.mask_radius_px = int(mask_radius_px)
        self.min_mask_px = int(min_mask_px)
        self.depth_mask_erode_px = int(depth_mask_erode_px)

        # State tracking
        self._samples: list[float] = []
        self._consecutive_na_count: int = 0
        self._last_completed_result: Optional[InspectionResult] = None

    def reset(self) -> None:
        """Reset sample window state."""
        self._samples.clear()
        self._consecutive_na_count = 0

    @classmethod
    def load_config(cls, cfg_path: str | Path) -> "DepthWindowInspector":
        """Load configuration from JSON file."""
        p = Path(cfg_path)
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                di = data.get("depth_inspect", data)
                hsv_range = di.get("hsv_blue", di.get("hsv", {}).get("blue", []))
                lo = tuple(hsv_range[0]) if len(hsv_range) >= 2 else (102, 175, 118)
                hi = tuple(hsv_range[1]) if len(hsv_range) >= 2 else (179, 255, 255)
                uv_raw = di.get("uv")
                uv_tuple = tuple(uv_raw) if uv_raw and len(uv_raw) >= 2 else None

                return cls(
                    ref_depth_mm=di.get("ref_depth_mm", 388.0) or 388.0,
                    tol_mm=di.get("tol_mm", 1.0),
                    kicker_num=di.get("kicker_num", 2),
                    min_samples=di.get("min_samples", 3),
                    idle_debounce_frames=di.get("idle_debounce_frames", 3),
                    use_center=di.get("use_center", True),
                    uv=uv_tuple,
                    sample_radius=di.get("depth_radius_px", di.get("sample_radius", 1)),
                    hsv_lo=lo,
                    hsv_hi=hi,
                    require_mask=di.get("require_mask", True),
                    mask_radius_px=di.get("mask_radius_px", 12),
                    min_mask_px=di.get("min_mask_px", 20),
                    depth_mask_erode_px=di.get("depth_mask_erode_px", 5),
                )
            except Exception:
                pass
        return cls()

    def resolve_uv(self, shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Resolve measurement UV: center of frame if use_center=True, else configured uv."""
        h, w = shape[:2]
        if self.use_center or self.uv is None:
            return (w // 2, h // 2)
        return self.uv

    def process_depth(
        self,
        depth_val_mm: Optional[float],
        uv: Optional[Tuple[int, int]] = None,
        mask_ok: bool = False,
    ) -> InspectionResult:
        """Process one depth measurement and update transit state."""
        # Case 1: Valid depth detected (object present under sensor)
        if depth_val_mm is not None:
            self._consecutive_na_count = 0
            self._samples.append(depth_val_mm)
            current_min = min(self._samples)
            return InspectionResult(
                event="MEASURING",
                status="MEASURE",
                ok=None,
                current_depth_mm=round(depth_val_mm, 1),
                min_depth_mm=round(current_min, 1),
                ref_depth_mm=self.ref_depth_mm,
                delta_mm=round(current_min - self.ref_depth_mm, 1),
                should_kick=False,
                kicker_num=self.kicker_num,
                samples_count=len(self._samples),
                uv=uv,
                mask_ok=mask_ok,
            )

        # Case 2: Depth is None (N/A / Empty)
        if len(self._samples) >= self.min_samples:
            self._consecutive_na_count += 1
            if self._consecutive_na_count >= self.idle_debounce_frames:
                # Window complete: calculate min depth over whole transit
                min_val = min(self._samples)
                delta = min_val - self.ref_depth_mm
                is_ok = abs(delta) <= self.tol_mm
                status = "OK" if is_ok else "NG"
                count = len(self._samples)

                result = InspectionResult(
                    event="OBJECT_DONE",
                    status=status,
                    ok=is_ok,
                    current_depth_mm=None,
                    min_depth_mm=round(min_val, 1),
                    ref_depth_mm=self.ref_depth_mm,
                    delta_mm=round(delta, 1),
                    should_kick=(not is_ok),
                    kicker_num=self.kicker_num,
                    samples_count=count,
                    uv=uv,
                    mask_ok=mask_ok,
                )

                self._samples.clear()
                self._consecutive_na_count = 0
                self._last_completed_result = result
                return result
        elif self._samples:
            self._consecutive_na_count += 1
            if self._consecutive_na_count >= self.idle_debounce_frames:
                self._samples.clear()
                self._consecutive_na_count = 0

        # Idle state
        return InspectionResult(
            event="IDLE",
            status="EMPTY",
            ok=None,
            current_depth_mm=None,
            min_depth_mm=None if not self._samples else round(min(self._samples), 1),
            ref_depth_mm=self.ref_depth_mm,
            delta_mm=None,
            should_kick=False,
            kicker_num=self.kicker_num,
            samples_count=len(self._samples),
            uv=uv,
            mask_ok=mask_ok,
        )

    def process_frame(
        self,
        depth_map: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
    ) -> InspectionResult:
        """Process depth & rgb frame using center point + HSV mask gating."""
        if depth_map is None or depth_map.size == 0:
            return self.process_depth(None, None, False)

        u, v = self.resolve_uv(depth_map.shape)
        mask_ok = False
        depth_mask = None

        if rgb_frame is not None and self.require_mask:
            mask = color_mask(rgb_frame, self.hsv_lo, self.hsv_hi)
            mask_ok = mask_hits_near(
                mask, u, v, radius=self.mask_radius_px, min_px=self.min_mask_px
            )
            if not mask_ok:
                # No blue object at center
                return self.process_depth(None, uv=(u, v), mask_ok=False)

            depth_mask = shrink_mask(mask, self.depth_mask_erode_px)

        d_mm = depth_at_uv(
            depth_map,
            u,
            v,
            radius=self.sample_radius,
            mask=depth_mask,
        )
        return self.process_depth(d_mm, uv=(u, v), mask_ok=mask_ok)


class ShmDepthReader:
    """Fast zero-copy / mmap reader for HP60C shared memory (/dev/shm/hp60c_frames)."""

    def __init__(self, path: str = SHM_PATH) -> None:
        self.path = path
        self._fd: Optional[int] = None
        self._buf: Optional[mmap.mmap] = None
        self._open_shm()

    def _open_shm(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            if self._buf is not None:
                self._buf.close()
            if self._fd is not None:
                os.close(self._fd)
            self._fd = os.open(self.path, os.O_RDONLY)
            self._buf = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        if self._buf is None:
            return self._open_shm()
        return True

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Read (rgb, depth, frame_id) from shared memory."""
        if not self.is_available() or self._buf is None:
            return None, None, 0

        try:
            magic = struct.unpack_from("<I", self._buf, 0)[0]
            if magic != SHM_MAGIC:
                return None, None, 0

            rgb_w, rgb_h = struct.unpack_from("<II", self._buf, 4)
            depth_w, depth_h = struct.unpack_from("<II", self._buf, 16)
            frame_id = struct.unpack_from("<Q", self._buf, 32)[0]
            rgb_ready = struct.unpack_from("<I", self._buf, 48)[0]
            depth_ready = struct.unpack_from("<I", self._buf, 52)[0]

            rgb = None
            if rgb_ready and rgb_w > 0 and rgb_h > 0:
                n = rgb_w * rgb_h * 3
                rgb = np.frombuffer(
                    self._buf, dtype=np.uint8, count=n, offset=HEADER_SIZE
                ).reshape(rgb_h, rgb_w, 3).copy()

            depth = None
            if depth_ready and depth_w > 0 and depth_h > 0:
                n = depth_w * depth_h
                offset = HEADER_SIZE + MAX_RGB_SIZE
                depth = np.frombuffer(
                    self._buf, dtype=np.uint16, count=n, offset=offset
                ).reshape(depth_h, depth_w).copy()

            return rgb, depth, frame_id
        except Exception:
            return None, None, 0

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
            self._buf = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def __enter__(self) -> "ShmDepthReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()
