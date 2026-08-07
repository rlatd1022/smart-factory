#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anomaly_detector.py — ResNet-18 기반 PatchCore 비지도 이상치 탐지, 결함 경향성 분석 및 360도 다중 결함 검출 모듈."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors


@dataclass
class DefectInfo:
    """개별 결함 클러스터 또는 X자 분기점 상세 정보."""
    defect_id: int  # 1-based index (1, 2, 3, ...)
    centroid_pix: Tuple[int, int]  # (u, v) in 256x256
    area_px: float  # 결함 면적 (픽셀)
    peak_score: float  # 해당 결함 내부 최고 이상치 점수
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in 256x256
    polar_coords: Tuple[float, float]  # (r_mm, theta_deg) 12시 기준점 대비 극좌표
    clock_pos: str  # 예: "12 o'clock", "2 o'clock", "7 o'clock"


@dataclass
class PatternAnalysisResult:
    """결함 패턴 형태 및 경향성 분석 결과."""
    severity: str  # "NORMAL", "MINOR", "MODERATE", "CRITICAL"
    pattern_type: str  # "CROSS_PATTERN", "LINEAR_SCRATCH", "POINT_CLUSTER", "EDGE_PERIMETER", "NONE"
    quadrant_density: Dict[str, float]  # {"Q1_NE": %, "Q2_SE": %, "Q3_SW": %, "Q4_NW": %}
    defect_area_ratio: float  # 원판 전체 면적 대비 결함 면적 (%)
    description: str  # 사람이 읽기 쉬운 패턴 요약 설명


@dataclass
class AnomalyResult:
    """이상치 검사 종합 결과."""
    is_defect: bool
    anomaly_score: float
    threshold: float
    heatmap: np.ndarray  # 256x256 float32
    overlay_bgr: np.ndarray  # 256x256 BGR 히트맵 및 결함 마킹 오버레이
    defects: List[DefectInfo] = field(default_factory=list)
    num_defects: int = 0
    dispersion_mm: float = 0.0  # 결함들의 공간적 산포도 (결함 간 평균 분산 거리 mm)
    pattern: Optional[PatternAnalysisResult] = None  # 패턴 경향성 분석 결과

    @property
    def defect_point_pix(self) -> Optional[Tuple[int, int]]:
        """하위 호환성: 최우선(가장 심각한) 결함의 중심 픽셀 좌표."""
        return self.defects[0].centroid_pix if self.defects else None

    @property
    def polar_coords(self) -> Optional[Tuple[float, float]]:
        """하위 호환성: 최우선 결함의 12시 기준 극좌표."""
        return self.defects[0].polar_coords if self.defects else None


class FeatureExtractor(nn.Module):
    """ResNet-18의 layer2와 layer3에서 다중 스케일 패치 특징을 추출하는 모델."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        f2 = self.layer2(x)  # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)

        # Layer 2와 Layer 3 패치 결합
        f2_pool = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3_pool = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)

        # f3를 f2 해상도(32x32)로 업샘플링 후 채널 결합
        f3_up = F.interpolate(f3_pool, size=f2_pool.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([f2_pool, f3_up], dim=1)  # (B, 384, 32, 32)
        return features


class PatchCoreDetector:
    """정상 샘플만으로 학습하여 360도 전구역 다중 결함 및 마커를 정밀 검출하고 경향성을 분석하는 탐지기."""

    def __init__(
        self,
        threshold: float = 0.5,
        wafer_radius_mm: float = 40.0,
        subsample_ratio: float = 0.1,
        mask_sticker_region: bool = True,
        min_defect_area_px: float = 8.0,
        mask_radius_ratio: float = 0.405,
    ):
        self.threshold = threshold
        self.wafer_radius_mm = wafer_radius_mm
        self.subsample_ratio = subsample_ratio
        self.mask_sticker_region = mask_sticker_region
        self.min_defect_area_px = min_defect_area_px
        self.mask_radius_ratio = mask_radius_ratio

        self.device = torch.device("cpu")
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.nn_model: Optional[NearestNeighbors] = None
        self.memory_bank: Optional[np.ndarray] = None
        self.mean_score: float = 0.0
        self.std_score: float = 0.0

        # ImageNet 정규화 파라미터
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    @staticmethod
    def _apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """LAB 색공간에서 L 채널에 CLAHE(적응형 히스토그램 균일화)를 적용하여 저조도 국소 대비를 극대화."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_ch)
        merged = cv2.merge((cl, a_ch, b_ch))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _preprocess(self, bgr_list: List[np.ndarray]) -> torch.Tensor:
        """256x256 BGR 이미지를 CLAHE 대비 강화 후 Tensor로 변환 및 정규화."""
        tensors = []
        for bgr in bgr_list:
            if bgr.shape[:2] != (256, 256):
                bgr = cv2.resize(bgr, (256, 256))
            enhanced_bgr = self._apply_clahe(bgr, clip_limit=2.0)
            rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - self.norm_mean) / self.norm_std
            chw = rgb.transpose(2, 0, 1)
            tensors.append(torch.from_numpy(chw))
        return torch.stack(tensors, dim=0).to(self.device)

    def _get_inspection_mask(self, size: int = 256) -> np.ndarray:
        """원형 판 내부 표면 영역 전체를 포괄하고 12시 마커만 정밀 마스킹."""
        mask = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        radius = int(size * self.mask_radius_ratio)
        cv2.circle(mask, (center, center), radius, 255, -1)

        if self.mask_sticker_region:
            # 12시 방향 상단 스티커 영역을 실제 스티커 크기에 맞게 정밀 마스킹
            sticker_cy = int(center - size * 0.33)
            cv2.circle(mask, (center, sticker_cy), int(size * 0.08), 0, -1)

        return mask

    def fit(self, normal_images: List[np.ndarray]) -> float:
        """정상 이미지들로부터 Memory Bank 구축 및 최적 임계값 산출."""
        if len(normal_images) == 0:
            raise ValueError("학습할 정상 이미지가 비어 있습니다.")

        print(f"[PatchCore] {len(normal_images)}장의 정상 이미지로부터 특징 벡터 추출 중...")
        self.feature_extractor.eval()
        features_list = []

        with torch.no_grad():
            batch_size = 16
            for i in range(0, len(normal_images), batch_size):
                batch = normal_images[i : i + batch_size]
                x = self._preprocess(batch)
                f = self.feature_extractor(x)  # (B, C, H, W)
                B, C, H, W = f.shape
                f_perm = f.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
                features_list.append(f_perm)

        all_features = np.vstack(features_list)
        n_total = all_features.shape[0]

        # 코어셋 서브샘플링 (최대 3000개)
        target_size = min(max(int(n_total * self.subsample_ratio), 500), 3000)
        if target_size < n_total:
            indices = np.random.choice(n_total, size=target_size, replace=False)
            self.memory_bank = all_features[indices]
        else:
            self.memory_bank = all_features

        print(f"[PatchCore] Memory Bank 구축 완료: {self.memory_bank.shape[0]}개 패치 벡터 (차원: {self.memory_bank.shape[1]})")

        # Nearest Neighbors 인덱스 빌드
        self.nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        self.nn_model.fit(self.memory_bank)

        # 정상 데이터셋에 대한 이상치 점수 통계 산출
        train_scores = []
        for img in normal_images:
            res = self.predict(img)
            train_scores.append(res.anomaly_score)

        self.mean_score = float(np.mean(train_scores))
        self.std_score = float(np.std(train_scores))
        max_train = float(np.max(train_scores))

        # 내부 순수 표면 기준 기본 임계값 설정 (정상 데이터 최대치 대비 30% 안전 여유율 또는 4시그마 적용)
        self.threshold = max(max_train * 1.30, self.mean_score + 4.0 * self.std_score)
        print(f"[PatchCore] 학습 완료! 정상 점수 (평균: {self.mean_score:.4f}, 최대: {max_train:.4f}) -> 권장 임계값: {self.threshold:.4f}")
        return self.threshold

    def _calc_polar(self, u: float, v: float, size: int = 256) -> Tuple[float, float, str]:
        """정렬 이미지 상의 좌표 (u, v)를 12시 기준 극좌표 (r_mm, theta_deg, clock_pos)로 변환."""
        center = size / 2.0
        dx = u - center
        dy = v - center
        pix_radius = center * 0.84
        r_mm = float((np.hypot(dx, dy) / max(pix_radius, 1.0)) * self.wafer_radius_mm)

        raw_deg = np.degrees(np.arctan2(dy, dx))
        theta_deg = float((raw_deg + 90.0) % 360.0)  # 12시=0도, 3시=90도, 6시=180도, 9시=270도

        clock_num = int(round(theta_deg / 30.0)) % 12
        if clock_num == 0:
            clock_num = 12
        clock_pos = f"{clock_num} o'clock"

        return r_mm, theta_deg, clock_pos

    def _analyze_pattern(
        self,
        defects: List[DefectInfo],
        defect_mask: np.ndarray,
        insp_mask: np.ndarray,
        anomaly_score: float,
    ) -> PatternAnalysisResult:
        """다중 결함들의 공간 분포, 면적비, 사분면 점유율을 종합하여 패턴 경향성을 진단."""
        valid_px_count = max(float(np.sum(insp_mask > 0)), 1.0)
        defect_px_count = float(np.sum(defect_mask > 0))
        defect_area_ratio = (defect_px_count / valid_px_count) * 100.0

        if not defects or anomaly_score < self.threshold:
            return PatternAnalysisResult(
                severity="NORMAL",
                pattern_type="NONE",
                quadrant_density={"Q1_NE": 0.0, "Q2_SE": 0.0, "Q3_SW": 0.0, "Q4_NW": 0.0},
                defect_area_ratio=0.0,
                description="양품 (이상 패턴 없음)",
            )

        # 사분면별 결함 픽셀 분포 계산 (중심 128, 128 기준)
        cy, cx = 128, 128
        y_coords, x_coords = np.where(defect_mask > 0)
        total_defect_px = max(len(x_coords), 1)

        q1 = np.sum((x_coords >= cx) & (y_coords < cy))   # NE (12~3시)
        q2 = np.sum((x_coords >= cx) & (y_coords >= cy))  # SE (3~6시)
        q3 = np.sum((x_coords < cx) & (y_coords >= cy))   # SW (6~9시)
        q4 = np.sum((x_coords < cx) & (y_coords < cy))    # NW (9~12시)

        quadrant_density = {
            "Q1_NE": round((q1 / total_defect_px) * 100.0, 1),
            "Q2_SE": round((q2 / total_defect_px) * 100.0, 1),
            "Q3_SW": round((q3 / total_defect_px) * 100.0, 1),
            "Q4_NW": round((q4 / total_defect_px) * 100.0, 1),
        }

        # 4개 사분면 중 유의미한 결함(>10%)이 몇 개 사분면에 걸쳐있는지 카운트
        active_quadrants = sum(1 for v in quadrant_density.values() if v >= 10.0)

        # 결함 패턴 형태 분류
        num_def = len(defects)
        max_radius = max(d.polar_coords[0] for d in defects) if defects else 0.0

        if active_quadrants >= 3 or num_def >= 4 or defect_area_ratio >= 3.0:
            pattern_type = "CROSS_PATTERN"
            pattern_desc = "X자/교차형 복합 대형 결함"
        elif max_radius >= 32.0 and defect_area_ratio < 2.0:
            pattern_type = "EDGE_PERIMETER"
            pattern_desc = "외곽 테두리 림 결함"
        elif defect_area_ratio >= 1.0 or num_def >= 2:
            pattern_type = "LINEAR_SCRATCH"
            pattern_desc = "선형 스크래치 / 마커 선 결함"
        else:
            pattern_type = "POINT_CLUSTER"
            pattern_desc = "국소 점형 / 스팟 미세 결함"

        # 심각도 등급 (Severity Grade) 결정
        if defect_area_ratio >= 3.0 or anomaly_score >= self.threshold * 2.0 or pattern_type == "CROSS_PATTERN":
            severity = "CRITICAL"
        elif defect_area_ratio >= 1.0 or anomaly_score >= self.threshold * 1.3:
            severity = "MODERATE"
        else:
            severity = "MINOR"

        full_desc = f"{pattern_desc} [심각도: {severity}, 결함면적: {defect_area_ratio:.2f}%]"

        return PatternAnalysisResult(
            severity=severity,
            pattern_type=pattern_type,
            quadrant_density=quadrant_density,
            defect_area_ratio=round(defect_area_ratio, 2),
            description=full_desc,
        )

    def predict(self, aligned_bgr: np.ndarray) -> AnomalyResult:
        """12시 정렬 이미지에서 360도 전구역 결함, X자 패턴 및 경향성을 정밀 검출."""
        if self.nn_model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. 먼저 fit()을 호출하거나 load()하세요.")

        with torch.no_grad():
            x = self._preprocess([aligned_bgr])
            f = self.feature_extractor(x)
            _, C, H, W = f.shape
            f_flat = f.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

        distances, _ = self.nn_model.kneighbors(f_flat)
        dist_map = distances.reshape(H, W)

        # 256x256으로 리사이즈 및 가우시안 블러 스무딩
        heatmap = cv2.resize(dist_map, (256, 256), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), sigmaX=2.0)

        # 검사 유효 마스크 적용
        insp_mask = self._get_inspection_mask(256)
        heatmap[insp_mask == 0] = 0.0

        # 전체 최고 이상치 점수 (정상 표면 기준 안정적 이상치 평가)
        anomaly_score = float(np.max(heatmap))
        raw_is_defect = anomaly_score >= self.threshold

        defect_candidates = []
        defect_mask = np.zeros_like(insp_mask)

        if raw_is_defect:
            defect_mask[(heatmap >= self.threshold) & (insp_mask > 0)] = 255
            # 선형 결함 단절 방지를 위한 3x3 닫힘 연산
            kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kern_close)

            cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = float(cv2.contourArea(c))
                peri = float(cv2.arcLength(c, False))
                # 미세 스크래치/마커 보존 (둘레가 8 이상이거나 면적이 4 이상이면 검출)
                if area < 4.0 and peri < 8.0:
                    continue

                bx, by, bw, bh = cv2.boundingRect(c)
                c_mask = np.zeros_like(defect_mask)
                cv2.drawContours(c_mask, [c], -1, 255, -1)

                # X자와 같이 크거나 긴 복합 패턴인 경우: 4개 분기점/끝단과 중심 분할
                if bw > 28 or bh > 28:
                    corners = cv2.goodFeaturesToTrack(c_mask, maxCorners=6, qualityLevel=0.08, minDistance=15)
                    if corners is not None and len(corners) >= 2:
                        for pt in corners:
                            pu, pv = int(round(pt[0][0])), int(round(pt[0][1]))
                            if 0 <= pu < 256 and 0 <= pv < 256 and insp_mask[pv, pu] > 0:
                                p_val = float(heatmap[pv, pu])
                                r_mm, theta_deg, clock_pos = self._calc_polar(pu, pv, size=256)
                                defect_candidates.append({
                                    "centroid": (pu, pv),
                                    "area": max(area / len(corners), 1.0),
                                    "peak_score": p_val,
                                    "bbox": (max(pu - 8, 0), max(pv - 8, 0), 16, 16),
                                    "polar": (r_mm, theta_deg),
                                    "clock": clock_pos,
                                    "contour": c,
                                })
                        continue

                # 단일 스팟/결함 처리
                peak_val = float(np.max(heatmap[c_mask == 255])) if np.any(c_mask == 255) else anomaly_score
                M = cv2.moments(c)
                if M["m00"] > 1e-4:
                    cu = int(round(M["m10"] / M["m00"]))
                    cv = int(round(M["m01"] / M["m00"]))
                else:
                    cu = bx + bw // 2
                    cv = by + bh // 2

                r_mm, theta_deg, clock_pos = self._calc_polar(cu, cv, size=256)
                defect_candidates.append({
                    "centroid": (cu, cv),
                    "area": area,
                    "peak_score": peak_val,
                    "bbox": (bx, by, bw, bh),
                    "polar": (r_mm, theta_deg),
                    "clock": clock_pos,
                    "contour": c,
                })

        # 결함 심각도(Peak Score) 순으로 정렬하여 #1, #2, #3 부여
        defect_candidates.sort(key=lambda d: d["peak_score"], reverse=True)

        defects: List[DefectInfo] = []
        for idx, d in enumerate(defect_candidates, start=1):
            defects.append(
                DefectInfo(
                    defect_id=idx,
                    centroid_pix=d["centroid"],
                    area_px=d["area"],
                    peak_score=d["peak_score"],
                    bbox=d["bbox"],
                    polar_coords=d["polar"],
                    clock_pos=d["clock"],
                )
            )

        # 이상치 점수가 임계값을 넘었으나 분할에서 미세했던 경우 최댓점 위치 보존
        if raw_is_defect and len(defects) == 0:
            _, _, _, max_loc = cv2.minMaxLoc(heatmap)
            r_mm, theta_deg, clock_pos = self._calc_polar(max_loc[0], max_loc[1], size=256)
            defects.append(
                DefectInfo(
                    defect_id=1,
                    centroid_pix=max_loc,
                    area_px=8.0,
                    peak_score=anomaly_score,
                    bbox=(max(max_loc[0] - 5, 0), max(max_loc[1] - 5, 0), 10, 10),
                    polar_coords=(r_mm, theta_deg),
                    clock_pos=clock_pos,
                )
            )

        # 실제 유효 결함 존재 여부로 최종 판정
        is_defect = (len(defects) > 0) or raw_is_defect

        # 공간 산포도 계산
        dispersion_mm = 0.0
        if len(defects) >= 2:
            pts = np.array([d.centroid_pix for d in defects], dtype=np.float32)
            scale_mm = self.wafer_radius_mm / (128.0 * 0.84)
            pts_mm = pts * scale_mm
            mean_pt = np.mean(pts_mm, axis=0)
            dispersion_mm = float(np.mean(np.linalg.norm(pts_mm - mean_pt, axis=1)))

        # 패턴 경향성 분석 수행
        pattern_res = self._analyze_pattern(defects, defect_mask, insp_mask, anomaly_score)

        # 오버레이 이미지 생성 (Jet Colormap)
        norm_map = np.clip(heatmap / max(self.threshold * 1.5, 1e-4), 0.0, 1.0)
        norm_map_u8 = (norm_map * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_map_u8, cv2.COLORMAP_JET)
        color_map[insp_mask == 0] = 0

        overlay = cv2.addWeighted(aligned_bgr, 0.60, color_map, 0.40, 0)

        # 모든 검출된 다중 결함들에 번호 및 마킹 표시
        for d in defects:
            du, dv = d.centroid_pix
            bx, by, bw, bh = d.bbox
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
            cv2.drawMarker(overlay, (du, dv), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
            badge_text = f"#{d.defect_id}"
            cv2.putText(
                overlay,
                badge_text,
                (bx, max(by - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return AnomalyResult(
            is_defect=is_defect,
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            heatmap=heatmap,
            overlay_bgr=overlay,
            defects=defects,
            num_defects=len(defects),
            dispersion_mm=dispersion_mm,
            pattern=pattern_res,
        )

    def save(self, file_path: str):
        """학습된 Memory Bank와 가중치를 파일로 저장."""
        out_p = Path(file_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "memory_bank": self.memory_bank,
            "threshold": self.threshold,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "wafer_radius_mm": self.wafer_radius_mm,
            "mask_sticker_region": self.mask_sticker_region,
            "min_defect_area_px": self.min_defect_area_px,
            "mask_radius_ratio": self.mask_radius_ratio,
        }
        with open(out_p, "wb") as f:
            pickle.dump(data, f)
        print(f"[PatchCore] 모델 저장 완료 -> {out_p}")

    def load(self, file_path: str):
        """저장된 Memory Bank와 파라미터 로드."""
        in_p = Path(file_path)
        if not in_p.is_file():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {in_p}")
        with open(in_p, "rb") as f:
            data = pickle.load(f)
        self.memory_bank = data["memory_bank"]
        self.threshold = data["threshold"]
        self.mean_score = data.get("mean_score", 0.0)
        self.std_score = data.get("std_score", 0.0)
        self.wafer_radius_mm = data.get("wafer_radius_mm", 40.0)
        self.mask_sticker_region = data.get("mask_sticker_region", True)
        self.min_defect_area_px = data.get("min_defect_area_px", 3.0)
        self.mask_radius_ratio = data.get("mask_radius_ratio", 0.385)

        self.nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        self.nn_model.fit(self.memory_bank)
        print(f"[PatchCore] 모델 로드 완료 <- {in_p} (임계값: {self.threshold:.4f})")
