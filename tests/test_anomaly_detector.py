#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_anomaly_detector.py — 360도 전구역 다중 결함 이상치 탐지 엔진 단위 테스트."""

import cv2
import numpy as np
import pytest

from iotdemo.deeplearning.anomaly_detector import PatchCoreDetector


def generate_synthetic_normal_sample(size: int = 256) -> np.ndarray:
    """정상 파란색 원형 판 (12시 상단에 빨간색 스티커 부착) 합성 이미지 생성."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = int(size * 0.42)

    # 파란색 원형 판
    cv2.circle(img, (center, center), radius, (180, 80, 30), -1)

    # 정상 패턴 (중심부에 작은 동심원 패턴)
    cv2.circle(img, (center, center), int(radius * 0.4), (200, 110, 50), 2)

    # 12시 방향 빨간색 스티커 (cy=36, radius=18)
    sticker_cy = int(center - size * 0.36)
    cv2.circle(img, (center, sticker_cy), 18, (0, 0, 220), -1)

    # 미세 노이즈
    noise = np.random.normal(0, 2, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def test_patchcore_fit_and_detect(tmp_path):
    # 1. 정상 샘플 10장 생성
    normal_images = [generate_synthetic_normal_sample() for _ in range(10)]

    detector = PatchCoreDetector(
        wafer_radius_mm=40.0,
        subsample_ratio=0.2,
        mask_sticker_region=True,
    )

    # 2. 학습
    threshold = detector.fit(normal_images)
    assert threshold > 0.0
    assert detector.memory_bank is not None

    # 3. 정상 샘플 추론 -> OK 판정이어야 함
    test_normal = generate_synthetic_normal_sample()
    res_normal = detector.predict(test_normal)
    assert res_normal.is_defect is False
    assert res_normal.anomaly_score < threshold
    assert res_normal.num_defects == 0

    # 4. 8개 방향(0, 45, 90, 135, 180, 225, 270, 315도) 전방위 결함 검출 검증
    # 12시(0도)부터 9시(270도)까지 모든 위치에서 결함이 감지되는지 확인
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle in test_angles:
        test_defect_img = test_normal.copy()
        # 반경 20mm (~55px) 지점에 검은색 결함 점 추가
        rad = np.radians(angle - 90.0)
        px = int(128 + 55 * np.cos(rad))
        py = int(128 + 55 * np.sin(rad))
        cv2.circle(test_defect_img, (px, py), 6, (0, 0, 0), -1)

        res_angle = detector.predict(test_defect_img)
        assert res_angle.is_defect is True, f"각도 {angle}도(위치: {px},{py})에서 결함 미검출"
        assert res_angle.num_defects >= 1, f"각도 {angle}도에서 결함 개수 0"

    # 5. X자 복합 패턴 4개 방향 분기점 동시 검출 테스트
    test_x = test_normal.copy()
    cv2.line(test_x, (85, 85), (170, 170), (10, 10, 10), 6)
    cv2.line(test_x, (85, 170), (170, 85), (10, 10, 10), 6)

    res_x = detector.predict(test_x)
    assert res_x.is_defect is True
    assert res_x.num_defects >= 4, f"X자 분기점 개수 부족: {res_x.num_defects}"
    assert res_x.dispersion_mm > 0.0

    # 6. 모델 저장 및 로드
    model_file = tmp_path / "patchcore_360.pkl"
    detector.save(str(model_file))
    loaded = PatchCoreDetector()
    loaded.load(str(model_file))
    res_loaded = loaded.predict(test_x)
    assert res_loaded.is_defect is True
    assert res_loaded.num_defects == res_x.num_defects
