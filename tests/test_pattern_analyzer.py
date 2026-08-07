#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_pattern_analyzer.py — 결함 패턴 경향성 분석 및 하드웨어 속도 제어 단위 테스트."""

import cv2
import numpy as np
import pytest

from iotdemo.deeplearning.anomaly_detector import PatchCoreDetector
from iotdemo.factory_controller import FactoryController


def generate_wafer_plate(size: int = 256) -> np.ndarray:
    """테스트용 파란색 원형 판 생성."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    cv2.circle(img, (center, center), int(size * 0.42), (180, 80, 30), -1)
    # 12시 스티커
    cv2.circle(img, (center, int(center - size * 0.36)), 18, (0, 0, 220), -1)
    return img


def test_pattern_analysis_normal():
    detector = PatchCoreDetector(subsample_ratio=0.2)
    normals = [generate_wafer_plate() for _ in range(5)]
    detector.fit(normals)

    res = detector.predict(generate_wafer_plate())
    assert res.is_defect is False
    assert res.pattern is not None
    assert res.pattern.severity == "NORMAL"
    assert res.pattern.pattern_type == "NONE"


def test_pattern_analysis_cross_pattern():
    detector = PatchCoreDetector(subsample_ratio=0.2)
    normals = [generate_wafer_plate() for _ in range(5)]
    detector.fit(normals)

    # X자 마커 결함 추가
    x_img = generate_wafer_plate()
    cv2.line(x_img, (80, 80), (175, 175), (10, 10, 10), 8)
    cv2.line(x_img, (80, 175), (175, 80), (10, 10, 10), 8)

    res = detector.predict(x_img)
    assert res.is_defect is True
    assert res.pattern is not None
    assert res.pattern.pattern_type == "CROSS_PATTERN"
    assert res.pattern.severity == "CRITICAL"
    assert res.pattern.defect_area_ratio > 2.0


def test_pattern_analysis_linear_scratch():
    detector = PatchCoreDetector(subsample_ratio=0.2)
    normals = [generate_wafer_plate() for _ in range(5)]
    detector.fit(normals)

    # 단일 선형 스크래치 결함 추가
    line_img = generate_wafer_plate()
    cv2.line(line_img, (110, 120), (160, 150), (20, 20, 20), 4)

    res = detector.predict(line_img)
    assert res.is_defect is True
    assert res.pattern is not None
    assert res.pattern.pattern_type in ("LINEAR_SCRATCH", "CROSS_PATTERN")
    assert res.pattern.severity in ("MODERATE", "CRITICAL")


def test_factory_controller_conveyor_speed():
    # Dummy controller mode
    ctrl = FactoryController(conn=FactoryController.Connector.AUTO, port="dummy_test", debug=False)
    ctrl.conveyor_speed = 180
    assert ctrl.conveyor_speed == 180
    ctrl.set_conveyor_speed(220)
    assert ctrl.conveyor_speed == 220
