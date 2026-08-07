#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rule_detector.py — 12시 정렬 ROI 기반 룰베이스(HSV/윤곽선/마커) 초고속 결함 검출 엔진."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DefectCandidate:
    contour: np.ndarray
    area: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]  # (u, v) in 256x256
    polar_r_mm: float
    polar_theta_deg: float
    defect_type: str  # "MARKER_X", "SCRATCH", "FOREIGN_SPOT"


@dataclass
class RuleDefectResult:
    is_defect: bool
    defect_score: float  # 총 결함 픽셀 면적 (px)
    defect_count: int
    defects: List[DefectCandidate] = field(default_factory=list)
    overlay_bgr: np.ndarray = field(default_factory=lambda: np.zeros((256, 256, 3), dtype=np.uint8))
    mask_bgr: np.ndarray = field(default_factory=lambda: np.zeros((256, 256, 3), dtype=np.uint8))
    description: str = "정상"


class RuleBasedDefectDetector:
    """12시 방향 정렬된 256x256 웨이퍼 ROI 대상 고정밀 룰베이스 결함 검출기."""

    def __init__(
        self,
        wafer_radius_mm: float = 40.0,
        roi_size: int = 256,
        inspect_radius_px: float = 85.0,  # 256x256 정렬 이미지 내 안전 검사 반경
        sticker_top_cut_y: int = 76,  # 12시 스티커 마스킹 Y 상한선
        sticker_x_range: Tuple[int, int] = (95, 161),  # 12시 스티커 마스킹 X 범위
        min_defect_area: float = 12.0,  # 최소 불량 면적 (픽셀)
        dark_v_thresh: int = 72,  # 마커 펜선 어두운 밝기 임계값
        dark_gray_thresh: int = 78,  # 마커 그레이스케일 임계값
    ):
        self.wafer_radius_mm = wafer_radius_mm
        self.roi_size = roi_size
        self.inspect_radius_px = inspect_radius_px
        self.sticker_top_cut_y = sticker_top_cut_y
        self.sticker_x_range = sticker_x_range
        self.min_defect_area = min_defect_area
        self.dark_v_thresh = dark_v_thresh
        self.dark_gray_thresh = dark_gray_thresh

        # 사전 연산된 검사 마스크
        self.insp_mask = self._build_inspection_mask()

    def _build_inspection_mask(self) -> np.ndarray:
        """12시 스티커 및 외곽 테두리를 제외한 순수 웨이퍼 표면 검사 마스크 생성."""
        mask = np.zeros((self.roi_size, self.roi_size), dtype=np.uint8)
        center = (self.roi_size // 2, self.roi_size // 2)

        # 1. 웨이퍼 원형 내부 활성화 (반경 85px)
        cv2.circle(mask, center, int(round(self.inspect_radius_px)), 255, -1)

        # 2. 12시 빨간색 기준점 스티커 영역 제외
        x1, x2 = self.sticker_x_range
        mask[: self.sticker_top_cut_y, x1:x2] = 0

        # 3. 중심 회전축 구멍/노이즈 보호 (반경 8px)
        cv2.circle(mask, center, 8, 0, -1)
        return mask

    def _calc_polar(self, u: float, v: float) -> Tuple[float, float]:
        """정렬 이미지 (u, v) 좌표로부터 12시 기준 극좌표 (r_mm, theta_deg) 산출."""
        center = self.roi_size / 2.0
        dx = u - center
        dy = v - center
        pix_dist = np.hypot(dx, dy)
        r_mm = (pix_dist / (self.roi_size * 0.42)) * self.wafer_radius_mm
        raw_deg = np.degrees(np.arctan2(dy, dx))
        theta_deg = (raw_deg + 90.0) % 360.0
        return float(r_mm), float(theta_deg)

    def inspect(self, aligned_bgr: np.ndarray) -> RuleDefectResult:
        """256x256 정렬된 웨이퍼 ROI를 검사하여 결함 여부 및 상세 정보 반환."""
        if aligned_bgr is None or aligned_bgr.shape[0] < self.roi_size or aligned_bgr.shape[1] < self.roi_size:
            return RuleDefectResult(
                is_defect=False,
                defect_score=0.0,
                defect_count=0,
                description="유효하지 않은 이미지",
            )

        if aligned_bgr.shape[:2] != (self.roi_size, self.roi_size):
            aligned_bgr = cv2.resize(aligned_bgr, (self.roi_size, self.roi_size))

        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        valid_mask = self.insp_mask > 0

        # --- 1. 검은색 마커 펜선 / X자 결함 검출 ---
        dark_mask = (v_ch < self.dark_v_thresh) & (gray < self.dark_gray_thresh) & valid_mask

        # --- 2. 비정상 색상 이물/얼룩 검출 (푸른색 판 외의 붉은색/노란색/초록색 마킹) ---
        abnormal_color = (
            ((h_ch < 75) | (h_ch > 145))
            & (s_ch > 65)
            & (v_ch > 60)
            & valid_mask
        )

        # --- 3. 고대비 흰색 스크래치 / 금속 찍힘 검출 ---
        white_scratch = (v_ch > 245) & (s_ch < 35) & valid_mask

        # --- 4. 결함 마스크 결합 및 모폴로지 정제 ---
        raw_defect = np.zeros((self.roi_size, self.roi_size), dtype=np.uint8)
        raw_defect[dark_mask | abnormal_color | white_scratch] = 255

        # 1~2픽셀 미세 센서 노이즈 제거 (Opening) & 결함 픽셀 응집 (Closing)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        defect_mask = cv2.morphologyEx(raw_defect, cv2.MORPH_OPEN, clean_kernel)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, clean_kernel)

        # --- 5. 결함 윤곽선(Contour) 분석 및 판정 ---
        cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defect_candidates: List[DefectCandidate] = []
        total_defect_area = 0.0

        overlay = aligned_bgr.copy()
        # 기준 검사 영역 가이드라인 (녹색 원)
        cv2.circle(overlay, (self.roi_size // 2, self.roi_size // 2), int(round(self.inspect_radius_px)), (0, 255, 0), 1)
        # 12시 스티커 마스크 영역 표시 (청록색 사각형)
        x1, x2 = self.sticker_x_range
        cv2.rectangle(overlay, (x1, 0), (x2, self.sticker_top_cut_y), (255, 255, 0), 1)

        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < self.min_defect_area:
                continue

            total_defect_area += area
            bx, by, bw, bh = cv2.boundingRect(c)

            M = cv2.moments(c)
            if M["m00"] > 1e-4:
                cu = int(round(M["m10"] / M["m00"]))
                cv = int(round(M["m01"] / M["m00"]))
            else:
                cu = bx + bw // 2
                cv = by + bh // 2

            r_mm, theta_deg = self._calc_polar(cu, cv)

            # 결함 유형 판별
            if np.any(dark_mask[by : by + bh, bx : bx + bw]):
                dtype = "MARKER_X"
            elif np.any(abnormal_color[by : by + bh, bx : bx + bw]):
                dtype = "FOREIGN_SPOT"
            else:
                dtype = "SCRATCH"

            defect_candidates.append(
                DefectCandidate(
                    contour=c,
                    area=area,
                    bbox=(bx, by, bw, bh),
                    center=(cu, cv),
                    polar_r_mm=r_mm,
                    polar_theta_deg=theta_deg,
                    defect_type=dtype,
                )
            )

            # 오버레이에 결함 표시 (빨간색 바운딩 박스 & 중심 마커)
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.drawContours(overlay, [c], -1, (0, 0, 255), 1)
            cv2.drawMarker(overlay, (cu, cv), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

            # 극좌표 텍스트 라벨 (예: "r:15mm, 120°")
            label = f"{r_mm:.0f}mm,{theta_deg:.0f}\xb0"
            cv2.putText(
                overlay,
                label,
                (max(bx - 10, 5), max(by - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        is_defect = len(defect_candidates) > 0
        mask_bgr = cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2BGR)

        if is_defect:
            desc = f"결함 {len(defect_candidates)}건 검출 (면적: {total_defect_area:.0f}px)"
            cv2.putText(
                overlay,
                f"NG: {len(defect_candidates)} DEFECT(S)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            desc = "정상 양품"
            cv2.putText(
                overlay,
                "OK: PASS (CLEAN)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return RuleDefectResult(
            is_defect=is_defect,
            defect_score=total_defect_area,
            defect_count=len(defect_candidates),
            defects=defect_candidates,
            overlay_bgr=overlay,
            mask_bgr=mask_bgr,
            description=desc,
        )
