#!/usr/bin/env python3
"""
iotdemo.vision.wafer_spatial_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
실시간 가상 원형 웨이퍼 공간 불량 확률 및 결함 집중 경향성(Hotspot) 분석기.

웨이퍼 결함의 위치(극좌표 및 직교좌표)를 실시간으로 누적하여:
1. 가상의 원형 웨이퍼 판 상의 2D 가우시안 결함 밀도 히트맵(Spatial Heatmap) 생성
2. 사분면(Q1~Q4) 및 반경별(Center, Mid, Edge) 불량 발생 확률(Probability %) 산출
3. 불량 다발 집중 영역(Hotspot) 및 공간적 패턴 자동 진단 문구 생성
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SpatialDefectPoint:
    """단일 결함 공간 기록 데이터."""
    object_id: int
    defect_type: str
    r_mm: float
    theta_deg: float
    cx_px: float
    cy_px: float
    area_px: float = 0.0
    timestamp_sec: float = 0.0


@dataclass
class SpatialAnalyticsSummary:
    """실시간 공간 결함 통계 및 경향성 분석 요약."""
    total_wafers: int
    defect_wafers: int
    total_defects: int
    overall_defect_rate: float  # %
    quadrant_prob: Dict[str, float]  # Q1_NE, Q2_SE, Q3_SW, Q4_NW (%)
    quadrant_counts: Dict[str, int]
    quadrant_shares: Dict[str, float]  # % of total defects
    radial_prob: Dict[str, float]  # CENTER, MID, EDGE (%)
    radial_counts: Dict[str, int]
    radial_shares: Dict[str, float]  # %
    top_hotspot_quadrant: str
    top_hotspot_prob: float  # %
    tendency_diagnosis: str
    is_hotspot_detected: bool


class WaferSpatialAnalyzer:
    """실시간 가상 원형 웨이퍼 공간 결함 밀도 및 확률 분석기."""

    def __init__(
        self,
        wafer_radius_mm: float = 45.0,
        grid_size: int = 256,
        max_history_points: int = 200,
        decay_factor: float = 1.0,  # 1.0 = 완전 누적, <1.0 = 최근 가중치 감쇠
    ):
        self.wafer_radius_mm = wafer_radius_mm
        self.grid_size = grid_size
        self.max_history_points = max_history_points
        self.decay_factor = decay_factor

        # 누적 카운터
        self.total_wafers = 0
        self.defect_wafers = 0
        self.total_defects = 0

        # 구역별 웨이퍼 불량 발생 횟수 (확률 분자용: 해당 구역에 결함이 나타난 웨이퍼 수)
        self.wafer_hits_quadrant = {"Q1_NE": 0, "Q2_SE": 0, "Q3_SW": 0, "Q4_NW": 0}
        self.wafer_hits_radial = {"CENTER": 0, "MID": 0, "EDGE": 0}

        # 구역별 총 결함 개수 (점유율 분자용)
        self.defect_counts_quadrant = {"Q1_NE": 0, "Q2_SE": 0, "Q3_SW": 0, "Q4_NW": 0}
        self.defect_counts_radial = {"CENTER": 0, "MID": 0, "EDGE": 0}

        # 8개 세부 섹터 (12시 기준 시계방향: N, NE, E, SE, S, SW, W, NW)
        self.sector_names = ["N_0", "NE_45", "E_90", "SE_135", "S_180", "SW_225", "W_270", "NW_315"]
        self.wafer_hits_sector = {s: 0 for s in self.sector_names}
        self.defect_counts_sector = {s: 0 for s in self.sector_names}

        # 결함 이력 리스트
        self.history_points: List[SpatialDefectPoint] = []

        # 2D 가우시안 누적 밀도 맵 (grid_size x grid_size)
        self.density_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # 웨이퍼 마스크 사전 생성
        self._disk_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._center_xy = (self.grid_size // 2, self.grid_size // 2)
        self._disk_radius_px = int(self.grid_size * 0.44)  # 여백을 둔 원형 판 반경
        cv2.circle(self._disk_mask, self._center_xy, self._disk_radius_px, 255, -1)

    def reset(self):
        """누적 통계 및 2D 히트맵 초기화."""
        self.total_wafers = 0
        self.defect_wafers = 0
        self.total_defects = 0
        self.wafer_hits_quadrant = {"Q1_NE": 0, "Q2_SE": 0, "Q3_SW": 0, "Q4_NW": 0}
        self.wafer_hits_radial = {"CENTER": 0, "MID": 0, "EDGE": 0}
        self.defect_counts_quadrant = {"Q1_NE": 0, "Q2_SE": 0, "Q3_SW": 0, "Q4_NW": 0}
        self.defect_counts_radial = {"CENTER": 0, "MID": 0, "EDGE": 0}
        self.wafer_hits_sector = {s: 0 for s in self.sector_names}
        self.defect_counts_sector = {s: 0 for s in self.sector_names}
        self.history_points.clear()
        self.density_map.fill(0.0)

    def add_pass(self):
        """양품(결함 없음) 웨이퍼 통과 기록."""
        self.total_wafers += 1

    def add_inspection_event(
        self,
        object_id: int,
        is_defect: bool,
        defects_info: List[dict],  # [{r_mm, theta_deg, defect_type, area_px, cx_px, cy_px}]
    ):
        """단일 웨이퍼 검사 이벤트 통합 등록."""
        self.total_wafers += 1
        if not is_defect or not defects_info:
            return

        self.defect_wafers += 1

        # 이번 웨이퍼에서 결함이 관측된 구역 집합 (웨이퍼당 1회만 카운트)
        touched_quadrants = set()
        touched_radials = set()
        touched_sectors = set()

        for d in defects_info:
            self.total_defects += 1
            r_mm = float(d.get("r_mm", 0.0))
            theta_deg = float(d.get("theta_deg", 0.0)) % 360.0
            defect_type = str(d.get("defect_type", "DEFECT"))
            area_px = float(d.get("area_px", 10.0))

            # 1) 사분면 판별 (12시=0도, 3시=90도, 6시=180도, 9시=270도)
            # 0~90: Q1(NE, 12~3시), 90~180: Q2(SE, 3~6시), 180~270: Q3(SW, 6~9시), 270~360: Q4(NW, 9~12시)
            if 0.0 <= theta_deg < 90.0:
                q_name = "Q1_NE"
            elif 90.0 <= theta_deg < 180.0:
                q_name = "Q2_SE"
            elif 180.0 <= theta_deg < 270.0:
                q_name = "Q3_SW"
            else:
                q_name = "Q4_NW"

            # 2) 반경 구역 판별
            if r_mm < 15.0:
                rad_name = "CENTER"
            elif r_mm < 35.0:
                rad_name = "MID"
            else:
                rad_name = "EDGE"

            # 3) 8섹터 판별 (45도 단위, 0도=12시 북쪽 기준)
            sec_idx = int(((theta_deg + 22.5) % 360.0) // 45.0)
            sec_name = self.sector_names[sec_idx % len(self.sector_names)]

            touched_quadrants.add(q_name)
            touched_radials.add(rad_name)
            touched_sectors.add(sec_name)

            self.defect_counts_quadrant[q_name] += 1
            self.defect_counts_radial[rad_name] += 1
            self.defect_counts_sector[sec_name] += 1

            # 4) 가상 캔버스 직교좌표 변환 (0도=12시 상단, 90도=3시 우측, 180도=6시 하단, 270도=9시 좌측)
            rad_norm = min(r_mm / max(self.wafer_radius_mm, 1e-3), 1.0)
            theta_rad = math.radians(theta_deg)
            px = self._center_xy[0] + rad_norm * self._disk_radius_px * math.sin(theta_rad)
            py = self._center_xy[1] - rad_norm * self._disk_radius_px * math.cos(theta_rad)

            point = SpatialDefectPoint(
                object_id=object_id,
                defect_type=defect_type,
                r_mm=r_mm,
                theta_deg=theta_deg,
                cx_px=px,
                cy_px=py,
                area_px=area_px,
            )
            self.history_points.append(point)
            if len(self.history_points) > self.max_history_points:
                self.history_points.pop(0)

            # 5) 2D 가우시안 스탬프 누적
            self._splat_gaussian(px, py, sigma=max(int(self.grid_size * 0.05), 4), weight=1.0)

        # 웨이퍼 기준 구역 히트수 누적
        for q in touched_quadrants:
            self.wafer_hits_quadrant[q] += 1
        for r in touched_radials:
            self.wafer_hits_radial[r] += 1
        for s in touched_sectors:
            self.wafer_hits_sector[s] += 1

    def _splat_gaussian(self, cx: float, cy: float, sigma: int = 12, weight: float = 1.0):
        """2D 가우시안 블러 커널을 밀도 맵에 누적."""
        gx = int(round(cx))
        gy = int(round(cy))
        ksize = sigma * 3
        x0, x1 = max(0, gx - ksize), min(self.grid_size, gx + ksize + 1)
        y0, y1 = max(0, gy - ksize), min(self.grid_size, gy + ksize + 1)
        if x0 >= x1 or y0 >= y1:
            return

        ys, xs = np.ogrid[y0 - gy : y1 - gy, x0 - gx : x1 - gx]
        dist_sq = xs * xs + ys * ys
        kernel = np.exp(-dist_sq / (2.0 * sigma * sigma)) * weight

        self.density_map[y0:y1, x0:x1] += kernel.astype(np.float32)

    def get_summary(self) -> SpatialAnalyticsSummary:
        """실시간 공간 불량 통계 및 경향성 분석 요약 데이터 반환."""
        n_wafers = max(self.total_wafers, 1)
        n_defects = max(self.total_defects, 1)
        overall_defect_rate = (self.defect_wafers / n_wafers) * 100.0

        quad_prob = {k: (self.wafer_hits_quadrant[k] / n_wafers) * 100.0 for k in self.wafer_hits_quadrant}
        quad_share = {k: (self.defect_counts_quadrant[k] / n_defects) * 100.0 for k in self.defect_counts_quadrant}

        rad_prob = {k: (self.wafer_hits_radial[k] / n_wafers) * 100.0 for k in self.wafer_hits_radial}
        rad_share = {k: (self.defect_counts_radial[k] / n_defects) * 100.0 for k in self.defect_counts_radial}

        # 최고 불량 발생 사분면 (Top Hotspot)
        top_quad = max(quad_prob.items(), key=lambda item: item[1])
        top_hotspot_quad = top_quad[0]
        top_hotspot_prob = top_quad[1]

        # 경향성 자동 진단 로직
        is_hotspot = False
        diagnosis_parts = []

        if self.total_defects == 0:
            tendency_str = "🟢 [정상] 누적 결함 없음 (청정 상태 유지)"
        else:
            # 사분면 편향도 검사 (최대 사분면 점유율이 45% 초과 시 편향 집중)
            top_share_item = max(quad_share.items(), key=lambda item: item[1])
            quad_labels = {"Q1_NE": "1사분면(NE)", "Q2_SE": "2사분면(SE)", "Q3_SW": "3사분면(SW)", "Q4_NW": "4사분면(NW)"}

            if top_share_item[1] >= 45.0 and self.total_defects >= 3:
                is_hotspot = True
                diagnosis_parts.append(f"⚠️ {quad_labels.get(top_share_item[0], top_share_item[0])}에 결함 {top_share_item[1]:.1f}% 집중")

            # 반경 방향 편향도 검사
            if rad_share["EDGE"] >= 55.0 and self.total_defects >= 3:
                is_hotspot = True
                diagnosis_parts.append(f"외곽 엣지(Edge Ring) 결함 다발({rad_share['EDGE']:.1f}%)")
            elif rad_share["CENTER"] >= 50.0 and self.total_defects >= 3:
                is_hotspot = True
                diagnosis_parts.append(f"중심부(Center Hub) 이물/얼룩 집중({rad_share['CENTER']:.1f}%)")

            if not diagnosis_parts:
                tendency_str = f"🟡 [균일 산포] 전 영역 무작위 결함 분포 (발생율: {overall_defect_rate:.1f}%)"
            else:
                tendency_str = f"🔴 [집중 경보] {' / '.join(diagnosis_parts)}"

        return SpatialAnalyticsSummary(
            total_wafers=self.total_wafers,
            defect_wafers=self.defect_wafers,
            total_defects=self.total_defects,
            overall_defect_rate=overall_defect_rate,
            quadrant_prob=quad_prob,
            quadrant_counts=self.defect_counts_quadrant.copy(),
            quadrant_shares=quad_share,
            radial_prob=rad_prob,
            radial_counts=self.defect_counts_radial.copy(),
            radial_shares=rad_share,
            top_hotspot_quadrant=top_hotspot_quad,
            top_hotspot_prob=top_hotspot_prob,
            tendency_diagnosis=tendency_str,
            is_hotspot_detected=is_hotspot,
        )

    def generate_wafer_map_image(self, target_size: Tuple[int, int] = (260, 260)) -> np.ndarray:
        """가상 원형 웨이퍼 판 위에 2D 가우시안 히트맵 및 결함 포인트를 렌더링한 BGR 이미지 생성."""
        tw, th = target_size
        img = np.full((th, tw, 3), (18, 22, 31), dtype=np.uint8)  # 배경 #12161F 다크 슬레이트

        cx = tw // 2
        cy = th // 2
        radius = int(min(tw, th) * 0.42)

        # 1. 2D 가우시안 밀도 히트맵 오버레이 (웨이퍼 원형 내부만)
        max_val = np.max(self.density_map)
        if max_val > 1e-4:
            # 밀도 정규화 (0~255)
            norm_density = np.clip(self.density_map * (255.0 / max_val), 0, 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)

            # 타겟 크기로 리사이즈
            heat_resized = cv2.resize(heat_color, (tw, th), interpolation=cv2.INTER_LINEAR)

            # 원형 마스크 생성
            mask = np.zeros((th, tw), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            # 알파 블렌딩 (가상 원형 웨이퍼 판과 히트맵 합성)
            alpha = 0.55
            blended = cv2.addWeighted(img, 1.0 - alpha, heat_resized, alpha, 0)
            img[mask == 255] = blended[mask == 255]

        # 2. 가상 원형 웨이퍼 기본 바디 그리기
        # 내부 음영 디스크
        disk_bg = np.zeros((th, tw, 3), dtype=np.uint8)
        cv2.circle(disk_bg, (cx, cy), radius, (35, 45, 60), -1)
        if max_val <= 1e-4:
            cv2.circle(img, (cx, cy), radius, (30, 38, 52), -1)

        # 3. 동심원 구역 가이드라인 (Center: 15mm, Mid: 35mm, Edge: 45mm)
        r_center = int(radius * (15.0 / self.wafer_radius_mm))
        r_mid = int(radius * (35.0 / self.wafer_radius_mm))

        cv2.circle(img, (cx, cy), r_center, (60, 80, 105), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r_mid, (60, 80, 105), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, (56, 189, 248), 2, cv2.LINE_AA)  # 외곽 엣지 (#38BDF8)

        # 4. 4사분면 십자 분할선 및 각도 라벨
        cv2.line(img, (cx - radius, cy), (cx + radius, cy), (60, 80, 105), 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - radius), (cx, cy + radius), (60, 80, 105), 1, cv2.LINE_AA)

        # 사분면 텍스트 라벨 (작은 폰트)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Q1", (cx + radius // 2 - 10, cy - radius // 2 + 10), font, 0.4, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(img, "Q2", (cx + radius // 2 - 10, cy + radius // 2 + 10), font, 0.4, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(img, "Q3", (cx - radius // 2 - 10, cy + radius // 2 + 10), font, 0.4, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(img, "Q4", (cx - radius // 2 - 10, cy - radius // 2 + 10), font, 0.4, (148, 163, 184), 1, cv2.LINE_AA)

        # 5. 12시 방향 정렬 마커/노치 표기 (상단)
        notch_pt1 = (cx, cy - radius - 5)
        notch_pt2 = (cx - 6, cy - radius + 6)
        notch_pt3 = (cx + 6, cy - radius + 6)
        cv2.fillPoly(img, [np.array([notch_pt1, notch_pt2, notch_pt3], dtype=np.int32)], (245, 158, 11))  # 주황색 노치
        cv2.putText(img, "12H", (cx - 10, cy - radius - 8), font, 0.35, (245, 158, 11), 1, cv2.LINE_AA)

        # 6. 최근 결함 산포 포인트 (Scatter Points) 그리기
        type_colors = {
            "MARKER_X": (236, 72, 153),    # 핫핑크/자주 (BGR)
            "FOREIGN_SPOT": (59, 130, 246), # 오렌지/레드 (BGR)
            "SCRATCH": (248, 189, 56),      # 청록/하늘 (BGR)
            "AI_ANOMALY": (79, 70, 229),    # 산호색
        }

        scale_x = tw / float(self.grid_size)
        scale_y = th / float(self.grid_size)

        for pt in self.history_points[-60:]:  # 최근 60개 결함 포인트 렌더링
            # grid_size 좌표를 타겟 해상도 좌표로 매핑
            sx = int(pt.cx_px * scale_x)
            sy = int(pt.cy_px * scale_y)
            color = type_colors.get(pt.defect_type, (0, 255, 255))

            # 광륜 효과 및 중심점
            cv2.circle(img, (sx, sy), 5, color, 1, cv2.LINE_AA)
            cv2.circle(img, (sx, sy), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 7. 중앙 축 허브 마커
        cv2.circle(img, (cx, cy), 3, (148, 163, 184), -1, cv2.LINE_AA)

        # 8. 좌하단 범례 및 정보
        summary = self.get_summary()
        stat_txt = f"Defects: {summary.total_defects} | Wafers: {summary.total_wafers}"
        cv2.putText(img, stat_txt, (10, th - 10), font, 0.38, (203, 213, 225), 1, cv2.LINE_AA)

        return img
