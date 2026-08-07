#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_wafer_aligner.py — WaferAligner 자동 회전 정렬 및 극좌표 변환 단위 테스트."""

import cv2
import numpy as np
import pytest

from iotdemo.vision.wafer_aligner import WaferAligner


def create_synthetic_wafer(angle_deg: float, size: int = 400) -> np.ndarray:
    """원형 파란색 판과 특정 각도에 위치한 빨간색 스티커 합성 이미지 생성.
    - angle_deg: 원형 중심 기준 스티커 각도 (0=3시, 90=6시, 180=9시, 270/-90=12시)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = 120

    # 파란색 원형 판 (BGR: 180, 50, 20 -> HSV Blue 영역)
    # HSV: (120, 220, 180) -> BGR로 변환
    hsv_blue = np.full((size, size, 3), (120, 220, 180), dtype=np.uint8)
    bgr_blue = cv2.cvtColor(hsv_blue, cv2.COLOR_HSV2BGR)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    img[mask == 255] = bgr_blue[mask == 255]

    # 빨간색 스티커 부착 (HSV: 0, 230, 220)
    rad = np.radians(angle_deg)
    sticker_r = radius * 0.82
    mx = int(round(cx + sticker_r * np.cos(rad)))
    my = int(round(cy + sticker_r * np.sin(rad)))

    hsv_red = np.full((1, 1, 3), (0, 230, 220), dtype=np.uint8)
    bgr_red = tuple(int(c) for c in cv2.cvtColor(hsv_red, cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(img, (mx, my), 8, bgr_red, -1)

    return img


def test_wafer_aligner_angles():
    aligner = WaferAligner(
        blue_lo=(100, 150, 100),
        blue_hi=(140, 255, 255),
        red_h1_hi=15,
        red_h2_lo=160,
        red_s_lo=50,
        red_v_lo=50,
        red_min_area=2.0,
        out_size=256,
    )

    test_angles = [0.0, 45.0, 90.0, 180.0, 270.0, -90.0]
    for angle in test_angles:
        frame = create_synthetic_wafer(angle)
        res = aligner.process(frame)

        assert res is not None, f"각도 {angle}도에서 정렬 실패"
        assert res.aligned_img.shape == (256, 256, 3)

        # 정렬된 이미지에서 빨간색 스티커가 12시 방향(상단)에 위치하는지 검증
        hsv_aligned = cv2.cvtColor(res.aligned_img, cv2.COLOR_BGR2HSV)
        red_mask = aligner.get_red_mask(hsv_aligned)
        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(cnts) > 0, f"정렬된 이미지에서 빨간색 스티커 검출 실패 (원래 각도: {angle})"

        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        mx = int(M["m10"] / M["m00"])
        my = int(M["m01"] / M["m00"])

        # 12시 방향 검증: X좌표는 중앙 부근(128 +- 10), Y좌표는 상단(128 미만)
        assert abs(mx - 128) < 15, f"12시 X좌표 오차 초과: {mx} (원래 각도 {angle})"
        assert my < 128, f"12시 Y좌표 상단 정렬 실패: {my} (원래 각도 {angle})"


def test_coordinate_mapping_and_polar():
    aligner = WaferAligner(out_size=256)
    frame = create_synthetic_wafer(45.0)
    res = aligner.process(frame)
    assert res is not None

    # 1. 256x256 중심점 -> 원본 중심점으로 역변환 매핑 검증
    orig_x, orig_y = res.aligned_to_original(128.0, 128.0)
    cx, cy = res.plate_info.center
    assert abs(orig_x - cx) < 2.0
    assert abs(orig_y - cy) < 2.0

    # 2. 극좌표 변환 검증 (12시 기준 90도 우측 = 3시 방향)
    # (128 + 50, 128) 지점은 3시 방향이므로 theta_deg ~ 90도
    r_mm, theta_deg = res.get_polar_coords(128.0 + 50.0, 128.0, wafer_radius_mm=40.0)
    assert r_mm > 0
    assert abs(theta_deg - 90.0) < 5.0
