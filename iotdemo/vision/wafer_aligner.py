#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wafer_aligner.py — 원형 판 및 기준점 스티커 검출, 12시 방향 자동 회전 정렬 및 좌표 변환 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PlateInfo:
    center: Tuple[int, int]
    radius: float
    contour: np.ndarray
    area: float


@dataclass
class MarkerInfo:
    centroid: Tuple[int, int]
    area: float
    angle_deg: float  # 원형 판 중심 기준 현재 스티커의 절대 각도 (도)


@dataclass
class AlignmentResult:
    aligned_img: np.ndarray  # 256x256 12시 방향 정렬된 이미지
    rot_deg: float  # 12시 방향으로 만들기 위해 회전한 각도
    matrix: np.ndarray  # 2x3 아핀 변환 행렬
    inv_matrix: np.ndarray  # 2x3 역 아핀 변환 행렬
    plate_info: PlateInfo
    marker_info: MarkerInfo

    def aligned_to_original(self, u: float, v: float) -> Tuple[float, float]:
        """정렬된 256x256 이미지 상의 좌표 (u, v)를 원본 카메라 영상 좌표 (x, y)로 변환."""
        pt = np.array([u, v, 1.0], dtype=np.float32)
        orig = self.inv_matrix @ pt
        return float(orig[0]), float(orig[1])

    def get_polar_coords(self, u: float, v: float, wafer_radius_mm: float = 40.0) -> Tuple[float, float]:
        """정렬된 이미지 상의 좌표 (u, v)로부터 12시 기준점 대비 극좌표 (r_mm, theta_deg) 산출.
        - r_mm: 중심으로부터의 물리적 거리 (mm)
        - theta_deg: 12시(북쪽=0도) 기준 시계방향 각도 (0 ~ 360도)
        """
        center_pix = self.aligned_img.shape[0] / 2.0
        dx = u - center_pix
        dy = v - center_pix

        pix_dist = np.hypot(dx, dy)
        r_mm = (pix_dist / max(self.plate_info.radius, 1.0)) * wafer_radius_mm

        raw_deg = np.degrees(np.arctan2(dy, dx))
        theta_deg = (raw_deg + 90.0) % 360.0
        return float(r_mm), float(theta_deg)


@dataclass
class WaferAlignedResult:
    """실시간 스트림 처리를 위한 종합 정렬 결과."""
    is_detected: bool
    aligned_roi: Optional[np.ndarray]  # 256x256 정렬 이미지
    overlay_view: np.ndarray  # 화면 표시용 가이드 오버레이 이미지
    alignment: Optional[AlignmentResult] = None


class WaferAligner:
    """원형 판 및 기준점 스티커를 감지하여 12시 방향으로 정규화 정렬하는 클래스."""

    def __init__(
        self,
        blue_lo: Tuple[int, int, int] = (102, 175, 118),
        blue_hi: Tuple[int, int, int] = (179, 255, 255),
        red_h1_hi: int = 15,
        red_h2_lo: int = 160,
        red_s_lo: int = 60,
        red_v_lo: int = 60,
        red_min_area: float = 2.0,
        plate_min_area: float = 300.0,
        out_size: int = 256,
    ):
        self.blue_lo = np.array(blue_lo, dtype=np.uint8)
        self.blue_hi = np.array(blue_hi, dtype=np.uint8)
        self.red_h1_hi = red_h1_hi
        self.red_h2_lo = red_h2_lo
        self.red_s_lo = red_s_lo
        self.red_v_lo = red_v_lo
        self.red_min_area = red_min_area
        self.plate_min_area = plate_min_area
        self.out_size = out_size

    def get_blue_mask(self, hsv: np.ndarray) -> np.ndarray:
        mask = cv2.inRange(hsv, self.blue_lo, self.blue_hi)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

    def get_red_mask(self, hsv: np.ndarray, roi_circle_mask: Optional[np.ndarray] = None) -> np.ndarray:
        lo1 = np.array([0, self.red_s_lo, self.red_v_lo], dtype=np.uint8)
        hi1 = np.array([self.red_h1_hi, 255, 255], dtype=np.uint8)
        lo2 = np.array([self.red_h2_lo, self.red_s_lo, self.red_v_lo], dtype=np.uint8)
        hi2 = np.array([180, 255, 255], dtype=np.uint8)

        m1 = cv2.inRange(hsv, lo1, hi1)
        m2 = cv2.inRange(hsv, lo2, hi2)
        mask = cv2.bitwise_or(m1, m2)

        if roi_circle_mask is not None:
            mask = cv2.bitwise_and(mask, mask, mask=roi_circle_mask)

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.dilate(mask, kern, iterations=1)

    def find_plate(self, hsv: np.ndarray) -> Optional[PlateInfo]:
        mask = self.get_blue_mask(hsv)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.plate_min_area:
            return None
        (x, y), radius = cv2.minEnclosingCircle(c)
        return PlateInfo(
            center=(int(round(x)), int(round(y))),
            radius=float(radius),
            contour=c,
            area=float(area),
        )

    def find_marker(self, hsv: np.ndarray, plate: PlateInfo) -> Optional[MarkerInfo]:
        cx, cy = plate.center
        radius = plate.radius

        roi_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(roi_mask, (cx, cy), int(radius * 1.1), 255, -1)

        red_mask = self.get_red_mask(hsv, roi_circle_mask=roi_mask)
        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_pt = None
        best_area = 0.0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.red_min_area or area > 4000.0:
                continue
            M = cv2.moments(c)
            if M["m00"] < 1e-4:
                bx, by, bw, bh = cv2.boundingRect(c)
                mx = int(bx + bw // 2)
                my = int(by + bh // 2)
            else:
                mx = int(round(M["m10"] / M["m00"]))
                my = int(round(M["m01"] / M["m00"]))

            dist = np.hypot(mx - cx, my - cy)
            if 0.1 * radius <= dist <= 1.15 * radius:
                if area >= best_area:
                    best_area = area
                    best_pt = (mx, my)

        if best_pt is None:
            return None

        mx, my = best_pt
        angle_deg = float(np.degrees(np.arctan2(my - cy, mx - cx)))
        return MarkerInfo(centroid=best_pt, area=best_area, angle_deg=angle_deg)

    def process(self, frame_bgr: np.ndarray) -> Optional[AlignmentResult]:
        """프레임에서 원형 판과 스티커를 찾아 12시 방향으로 회전 정렬된 결과 반환."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        plate = self.find_plate(hsv)
        if plate is None:
            return None

        marker = self.find_marker(hsv, plate)
        if marker is None:
            return None

        cx, cy = plate.center
        rot_deg = marker.angle_deg - (-90.0)
        scale = (self.out_size * 0.42) / max(plate.radius, 10.0)

        M = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale)
        M[0, 2] += (self.out_size / 2.0) - cx
        M[1, 2] += (self.out_size / 2.0) - cy

        M_full = np.vstack([M, [0, 0, 1]])
        M_inv_full = np.linalg.inv(M_full)
        M_inv = M_inv_full[:2, :]

        aligned = cv2.warpAffine(
            frame_bgr,
            M,
            (self.out_size, self.out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return AlignmentResult(
            aligned_img=aligned,
            rot_deg=rot_deg,
            matrix=M,
            inv_matrix=M_inv,
            plate_info=plate,
            marker_info=marker,
        )

    def align_frame(self, frame_bgr: np.ndarray) -> WaferAlignedResult:
        """실시간 프레임에 대해 오버레이 뷰와 정렬 ROI를 함께 생성하는 헬퍼 메서드."""
        overlay = frame_bgr.copy()
        res = self.process(frame_bgr)

        if res is None:
            cv2.putText(
                overlay,
                "WAITING FOR WAFER...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            return WaferAlignedResult(is_detected=False, aligned_roi=None, overlay_view=overlay, alignment=None)

        cx, cy = res.plate_info.center
        r = int(res.plate_info.radius)
        mx, my = res.marker_info.centroid

        # 판 외곽 원 및 중심점
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 2)
        cv2.drawMarker(overlay, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        # 12시 스티커 마커
        cv2.circle(overlay, (mx, my), 6, (0, 0, 255), -1)
        cv2.line(overlay, (cx, cy), (mx, my), (0, 255, 255), 1)

        cv2.putText(
            overlay,
            f"WAFER LOCKED (Rot: {res.rot_deg:+.1f} deg)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return WaferAlignedResult(
            is_detected=True,
            aligned_roi=res.aligned_img,
            overlay_view=overlay,
            alignment=res,
        )
