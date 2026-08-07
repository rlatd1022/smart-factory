#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_tracker_and_filter.py — 조명 불균일 억제, 단일 1회 검사 트래커, 속도 제어 단위 테스트."""

import time
import cv2
import numpy as np
import pytest

from iotdemo.factory_controller import FactoryController
from iotdemo.vision.wafer_aligner import WaferAligner, WaferAlignedResult, AlignmentResult, PlateInfo, MarkerInfo
from iotdemo.vision.wafer_tracker import WaferTransitTracker, TrackerState, TrackEvent, InspectionMode
from iotdemo.deeplearning.anomaly_detector import PatchCoreDetector


def _create_synthetic_wafer_with_shading(size=256, dark_marker=False) -> np.ndarray:
    """조명 그라디언트(한쪽이 어둡고 반대쪽이 밝은 음영)가 있는 가상 웨이퍼 생성."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.42)

    # 파란색 배경 판
    cv2.circle(img, center, radius, (140, 100, 30), -1)

    # 강한 조명 그라디언트 (X 방향으로 60단위 밝기 변화: 120 -> 180)
    for x in range(size):
        grad = int(np.interp(x, [0, size], [-30, +30]))
        col = np.clip(img[:, x].astype(np.int16) + grad, 0, 255).astype(np.uint8)
        img[:, x] = col

    # 12시 방향 빨간색 마커
    marker_pos = (center[0], center[1] - int(radius * 0.8))
    cv2.circle(img, marker_pos, 7, (20, 20, 200), -1)

    # 실제 불량인 경우에만 검은색 마커 X자 추가
    if dark_marker:
        cv2.line(img, (center[0] - 25, center[1] - 25), (center[0] + 25, center[1] + 25), (20, 20, 20), 4)
        cv2.line(img, (center[0] - 25, center[1] + 25), (center[0] + 25, center[1] - 25), (20, 20, 20), 4)

    return img


def test_shading_invariance_normal_wafer():
    """조명 그라디언트가 있어도 정상 웨이퍼는 오탐(False Positive)되지 않아야 함."""
    detector = PatchCoreDetector(threshold=4.0)
    # 가상 학습 이미지 5장
    train_imgs = [_create_synthetic_wafer_with_shading(256, dark_marker=False) for _ in range(5)]
    detector.fit(train_imgs)

    normal_shading_img = _create_synthetic_wafer_with_shading(256, dark_marker=False)
    res_normal = detector.predict(normal_shading_img)

    # Black-Hat 필터에 의해 완만한 조명 그라디언트가 제거되므로 정상으로 판정되어야 함
    assert not res_normal.is_defect or res_normal.num_defects == 0


def test_tracker_single_inspection_when_stationary():
    """웨이퍼가 컨베이어 정지로 화면 중앙에 멈춰 있어도 1회만 검사되어야 함."""
    tracker = WaferTransitTracker(center_tolerance_px=60.0, exit_debounce_frames=3, cooldown_sec=0.5)

    # 화면 중앙 (320, 240)에 위치한 웨이퍼 모의
    plate = PlateInfo(center=(320, 240), radius=100.0, contour=np.array([]), area=30000.0)
    marker = MarkerInfo(centroid=(320, 160), area=50.0, angle_deg=-90.0)
    align_mock = AlignmentResult(
        aligned_img=np.zeros((256, 256, 3), dtype=np.uint8),
        rot_deg=0.0,
        matrix=np.eye(2, 3),
        inv_matrix=np.eye(2, 3),
        plate_info=plate,
        marker_info=marker,
    )
    align_res = WaferAlignedResult(
        is_detected=True,
        aligned_roi=align_mock.aligned_img,
        overlay_view=np.zeros((480, 640, 3), dtype=np.uint8),
        alignment=align_mock,
    )

    inspect_triggers = 0
    # 50 프레임 동안 동일 위치 유지
    for _ in range(50):
        evt = tracker.update(align_res, (480, 640))
        if evt.should_inspect:
            inspect_triggers += 1

    # 정확히 단 1회만 트리거되어야 함 (중복 검사 0건)
    assert inspect_triggers == 1
    assert evt.state in (TrackerState.STOPPED_INSPECTING, TrackerState.INSPECTED)
    assert evt.object_id == 1

    tracker.notify_resumed()
    assert tracker.state == TrackerState.INSPECTED


def test_tracker_multiple_wafers_transit():
    """여러 웨이퍼가 차례대로 통과할 때 각각 1회씩 정상 검사되는지 검증."""
    tracker = WaferTransitTracker(center_tolerance_px=60.0, exit_debounce_frames=3, cooldown_sec=0.1)

    empty_res = WaferAlignedResult(
        is_detected=False,
        aligned_roi=None,
        overlay_view=np.zeros((480, 640, 3), dtype=np.uint8),
        alignment=None,
    )

    plate = PlateInfo(center=(320, 240), radius=100.0, contour=np.array([]), area=30000.0)
    marker = MarkerInfo(centroid=(320, 160), area=50.0, angle_deg=-90.0)
    align_mock = AlignmentResult(
        aligned_img=np.zeros((256, 256, 3), dtype=np.uint8),
        rot_deg=0.0,
        matrix=np.eye(2, 3),
        inv_matrix=np.eye(2, 3),
        plate_info=plate,
        marker_info=marker,
    )
    wafer_res = WaferAlignedResult(
        is_detected=True,
        aligned_roi=align_mock.aligned_img,
        overlay_view=np.zeros((480, 640, 3), dtype=np.uint8),
        alignment=align_mock,
    )

    total_inspections = 0

    # 3개의 웨이퍼가 진입 -> 정지/통과 -> 이탈 반복
    for _ in range(3):
        time.sleep(0.12)
        # 웨이퍼 진입 & 중심 통과
        for _ in range(10):
            evt = tracker.update(wafer_res, (480, 640))
            if evt.should_inspect:
                total_inspections += 1

        # 웨이퍼 이탈
        for _ in range(5):
            tracker.update(empty_res, (480, 640))

    assert total_inspections == 3
    assert tracker.current_object_id == 3


def test_conveyor_active_low_speed_control():
    """컨베이어 Active-LOW 상태 및 속도 설정 검증."""
    ctrl = FactoryController(conn=FactoryController.Connector.ARDUINO, port=None)

    # 가동 전: conveyor is False (DEV_OFF)
    assert not ctrl.conveyor

    # 시작: conveyor is True (DEV_ON)
    ctrl.system_start()
    assert ctrl.conveyor
    assert ctrl.green

    # 속도 변경
    ctrl.conveyor_speed = 180
    assert ctrl.conveyor_speed == 180

    # Stop-and-Inspect 제어용 conveyor_stop / conveyor_start 검증
    ctrl.conveyor_stop()
    assert not ctrl.conveyor

    ctrl.conveyor_start(200)
    assert ctrl.conveyor
    assert ctrl.conveyor_speed == 200

    ctrl.system_stop()
    assert not ctrl.conveyor


def test_stop_and_inspect_mode_sequence():
    """Stop-and-Inspect 모드에서 중심 도달 시 컨베이어 정지 트리거 발생 및 재가동 시퀀스 검증."""
    tracker = WaferTransitTracker(
        center_tolerance_px=60.0,
        exit_debounce_frames=3,
        cooldown_sec=0.1,
        mode=InspectionMode.STOP_AND_INSPECT,
    )

    plate = PlateInfo(center=(320, 240), radius=100.0, contour=np.array([]), area=30000.0)
    marker = MarkerInfo(centroid=(320, 160), area=50.0, angle_deg=-90.0)
    align_mock = AlignmentResult(
        aligned_img=np.zeros((256, 256, 3), dtype=np.uint8),
        rot_deg=0.0,
        matrix=np.eye(2, 3),
        inv_matrix=np.eye(2, 3),
        plate_info=plate,
        marker_info=marker,
    )
    wafer_res = WaferAlignedResult(
        is_detected=True,
        aligned_roi=align_mock.aligned_img,
        overlay_view=np.zeros((480, 640, 3), dtype=np.uint8),
        alignment=align_mock,
    )

    # 1. 중심 도달 시 Stop & Inspect 트리거 발생
    evt = tracker.update(wafer_res, (480, 640))
    assert evt.should_inspect is True
    assert evt.should_stop_conveyor is True
    assert evt.state == TrackerState.STOPPED_INSPECTING

    # 2. 정지 중 추가 프레임 입력 시 중복 검사 및 중복 정지 방지
    evt2 = tracker.update(wafer_res, (480, 640))
    assert evt2.should_inspect is False
    assert evt2.should_stop_conveyor is False

    # 3. 컨베이어 재가동 통지
    tracker.notify_resumed()
    assert tracker.state == TrackerState.INSPECTED


def test_tracker_mode_rollback_and_switch():
    """Stop-and-Inspect 모드와 Continuous(기존) 모드 간 실시간 롤백 및 전환 무결성 검증."""
    tracker = WaferTransitTracker(mode=InspectionMode.STOP_AND_INSPECT)
    assert tracker.mode == InspectionMode.STOP_AND_INSPECT

    # Continuous 모드로 롤백
    tracker.set_mode("CONTINUOUS")
    assert tracker.mode == InspectionMode.CONTINUOUS

    plate = PlateInfo(center=(320, 240), radius=100.0, contour=np.array([]), area=30000.0)
    marker = MarkerInfo(centroid=(320, 160), area=50.0, angle_deg=-90.0)
    align_mock = AlignmentResult(
        aligned_img=np.zeros((256, 256, 3), dtype=np.uint8),
        rot_deg=0.0,
        matrix=np.eye(2, 3),
        inv_matrix=np.eye(2, 3),
        plate_info=plate,
        marker_info=marker,
    )
    wafer_res = WaferAlignedResult(
        is_detected=True,
        aligned_roi=align_mock.aligned_img,
        overlay_view=np.zeros((480, 640, 3), dtype=np.uint8),
        alignment=align_mock,
    )

    # Continuous 모드에서는 should_stop_conveyor가 False여야 함
    evt = tracker.update(wafer_res, (480, 640))
    assert evt.should_inspect is True
    assert evt.should_stop_conveyor is False
    assert evt.state == TrackerState.INSPECTED

    # 다시 Stop-and-Inspect 모드로 전환
    tracker.set_mode(InspectionMode.STOP_AND_INSPECT)
    assert tracker.mode == InspectionMode.STOP_AND_INSPECT

