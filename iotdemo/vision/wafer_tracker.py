#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wafer_tracker.py — 컨베이어 위를 이동하는 웨이퍼의 단일 객체 추적 및 1회 검사 보장 상태 머신."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from iotdemo.vision.wafer_aligner import WaferAlignedResult


class InspectionMode(Enum):
    STOP_AND_INSPECT = "STOP_AND_INSPECT"  # 중심 도달 시 일시 정지 후 정밀 검사
    CONTINUOUS = "CONTINUOUS"              # 연속 이송 중 On-the-Fly 1회 검사 (기존 방식)


class TrackerState(Enum):
    IDLE = "IDLE"                          # 웨이퍼 없음 (진입 대기)
    APPROACHING = "APPROACHING"            # 웨이퍼 진입 후 중심 이동 중
    STOPPED_INSPECTING = "STOPPED_INSPECTING"  # 라인 정지 후 정밀 검사 수행 중
    INSPECTED = "INSPECTED"                # 1회 검사 완료 (라인 재가동 및 이탈 대기)
    DEPARTING = "DEPARTING"                # 화각 밖으로 이탈 중


@dataclass
class TrackEvent:
    should_inspect: bool                   # 이번 프레임에서 AI 비전 검사를 수행해야 하는지 여부
    should_stop_conveyor: bool             # 컨베이어 일시 정지 트리거 신호
    state: TrackerState                    # 현재 트래커 상태
    object_id: int                         # 단조 증가하는 개별 웨이퍼 고유 번호
    center_dist_px: float                  # 카메라 프레임 중심으로부터의 거리 (px)
    status_text: str                       # 디버그/화면 오버레이용 상태 문자열
    mode: InspectionMode = InspectionMode.STOP_AND_INSPECT


class WaferTransitTracker:
    """웨이퍼가 화각을 통과할 때 최적 위치에서 정확히 1회만 검사되도록 보장하는 상태 머신.
    
    STOP_AND_INSPECT 모드와 CONTINUOUS 모드를 지원하며, 언제든지 실시간 롤백/전환 가능합니다.
    """

    def __init__(
        self,
        center_tolerance_px: float = 65.0,
        exit_debounce_frames: int = 4,
        cooldown_sec: float = 0.8,
        mode: InspectionMode = InspectionMode.STOP_AND_INSPECT,
    ):
        self.center_tolerance_px = center_tolerance_px
        self.exit_debounce_frames = exit_debounce_frames
        self.cooldown_sec = cooldown_sec
        self.mode = mode

        self.state = TrackerState.IDLE
        self.current_object_id = 0
        self.miss_count = 0
        self.last_inspect_time = 0.0
        self.last_center: Optional[Tuple[int, int]] = None
        self.is_inspected_for_current = False

    def set_mode(self, mode: InspectionMode | str) -> None:
        """검사 모드 실시간 변경 (STOP_AND_INSPECT 또는 CONTINUOUS)."""
        if isinstance(mode, str):
            if mode.upper() in ("CONTINUOUS", "ON_THE_FLY", "FLOW"):
                self.mode = InspectionMode.CONTINUOUS
            else:
                self.mode = InspectionMode.STOP_AND_INSPECT
        else:
            self.mode = mode

    def notify_resumed(self) -> None:
        """컨베이어 재가동 후 검사 완료 상태로 전이."""
        if self.state == TrackerState.STOPPED_INSPECTING:
            self.state = TrackerState.INSPECTED

    def update(self, align_res: WaferAlignedResult, frame_shape: Tuple[int, int]) -> TrackEvent:
        """매 프레임마다 정렬 결과와 프레임 크기를 입력받아 검사 및 컨베이어 정지 트리거 판정."""
        fh, fw = frame_shape[:2]
        frame_cx, frame_cy = fw / 2.0, fh / 2.0
        now = time.time()

        # 1. 프레임 내에 웨이퍼가 감지되지 않은 경우
        if not align_res.is_detected or align_res.alignment is None:
            self.miss_count += 1
            if self.miss_count >= self.exit_debounce_frames:
                if self.state != TrackerState.IDLE:
                    self.state = TrackerState.IDLE
                    self.is_inspected_for_current = False
                    self.last_center = None

            return TrackEvent(
                should_inspect=False,
                should_stop_conveyor=False,
                state=self.state,
                object_id=self.current_object_id,
                center_dist_px=999.0,
                status_text=f"[{self.state.value}] WAITING FOR WAFER",
                mode=self.mode,
            )

        # 2. 웨이퍼가 감지된 경우
        self.miss_count = 0
        cx, cy = align_res.alignment.plate_info.center
        center_dist = float(np.hypot(cx - frame_cx, cy - frame_cy))

        # 새 웨이퍼 진입 감지
        if self.state == TrackerState.IDLE:
            self.current_object_id += 1
            self.state = TrackerState.APPROACHING
            self.is_inspected_for_current = False

        # 중심 허용 구역 내 도달 여부
        is_in_center_zone = center_dist <= self.center_tolerance_px

        should_inspect = False
        should_stop = False

        # 3. 1회 검사 트리거 조건:
        # - 아직 현재 객체에 대해 검사가 실행되지 않았음
        # - 중심 허용 구역에 도달함 (또는 첫 감지 시 중심 근처)
        # - 쿨다운 시간 충족
        if not self.is_inspected_for_current and (now - self.last_inspect_time >= self.cooldown_sec):
            if is_in_center_zone or center_dist < 100.0:
                should_inspect = True
                self.is_inspected_for_current = True
                self.last_inspect_time = now

                if self.mode == InspectionMode.STOP_AND_INSPECT:
                    should_stop = True
                    self.state = TrackerState.STOPPED_INSPECTING
                else:
                    self.state = TrackerState.INSPECTED

        self.last_center = (cx, cy)
        mode_tag = "STOP-INSPECT" if self.mode == InspectionMode.STOP_AND_INSPECT else "CONTINUOUS"
        status_str = f"[{self.state.value}|{mode_tag}] ID #{self.current_object_id} (Dist: {center_dist:.1f}px)"

        return TrackEvent(
            should_inspect=should_inspect,
            should_stop_conveyor=should_stop,
            state=self.state,
            object_id=self.current_object_id,
            center_dist_px=center_dist,
            status_text=status_str,
            mode=self.mode,
        )

    def reset(self):
        """트래커 상태 초기화."""
        self.state = TrackerState.IDLE
        self.current_object_id = 0
        self.miss_count = 0
        self.last_inspect_time = 0.0
        self.last_center = None
        self.is_inspected_for_current = False
