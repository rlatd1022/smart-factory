#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_hsv_mask.py — 실시간 HSV 마스크(파란색 원형 판 + 빨간색 기준 스티커) 검증 및 정렬 뷰어.

실행:
    cd /home/intel/smart_factory_edu
    python check_hsv_mask.py

기능:
  - 1. 원본 화면: 파란색 원형 판 외곽선(Green) + 빨간색 스티커(Red) + 12시 방향 정렬 벡터 실시간 오버레이
  - 2. 파란색 마스크 뷰: 파란색 원형 판 영역이 선명하게 하얗게 분리되는지 확인
  - 3. 빨간색 마스크 뷰: 빨간색 스티커 점이 정확하게 하얗게 분리되는지 확인
  - 4. 12시 정렬 ROI 뷰: 실시간으로 회전 정렬된 256x256 원형 샘플 미리보기
  - 5. 트랙바: 현장 조명에 따라 HSV 값을 실시간 미세조정 가능
  - 단축키:
      's': 현재 튜닝된 HSV 값을 depth_cfg.json에 저장
      'q' 또는 ESC: 종료
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# smart_factory_edu root
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from iotdemo.depth.depth_window_inspector import ShmDepthReader

# 기본 HSV 값
DEFAULT_BLUE_LO = (102, 175, 118)
DEFAULT_BLUE_HI = (179, 255, 255)

DEFAULT_RED_LO1 = (0, 100, 100)
DEFAULT_RED_HI1 = (10, 255, 255)
DEFAULT_RED_LO2 = (165, 100, 100)
DEFAULT_RED_HI2 = (180, 255, 255)

WIN_MAIN = "1. Camera Overlay & Live Vector"
WIN_MASKS = "2. HSV Masks (Left: Blue Plate, Right: Red Sticker)"
WIN_ALIGNED = "3. 12 O'Clock Aligned ROI (256x256)"


def nothing(x):
    pass


def setup_trackbars(
    blue_lo: Tuple[int, int, int],
    blue_hi: Tuple[int, int, int],
    red_s_lo: int = 60,
    red_v_lo: int = 60,
):
    cv2.namedWindow(WIN_MASKS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MASKS, 800, 450)

    # Blue Trackbars
    cv2.createTrackbar("Blue H Lo", WIN_MASKS, blue_lo[0], 180, nothing)
    cv2.createTrackbar("Blue H Hi", WIN_MASKS, blue_hi[0], 180, nothing)
    cv2.createTrackbar("Blue S Lo", WIN_MASKS, blue_lo[1], 255, nothing)
    cv2.createTrackbar("Blue S Hi", WIN_MASKS, blue_hi[1], 255, nothing)
    cv2.createTrackbar("Blue V Lo", WIN_MASKS, blue_lo[2], 255, nothing)
    cv2.createTrackbar("Blue V Hi", WIN_MASKS, blue_hi[2], 255, nothing)

    # Red Trackbars (대폭 확장)
    cv2.createTrackbar("Red H1 Hi", WIN_MASKS, 15, 30, nothing)
    cv2.createTrackbar("Red H2 Lo", WIN_MASKS, 160, 180, nothing)
    cv2.createTrackbar("Red S Lo", WIN_MASKS, red_s_lo, 255, nothing)
    cv2.createTrackbar("Red V Lo", WIN_MASKS, red_v_lo, 255, nothing)
    cv2.createTrackbar("Red Min Area", WIN_MASKS, 2, 50, nothing)


def get_blue_mask(hsv: np.ndarray, lo: Tuple[int, int, int], hi: Tuple[int, int, int]) -> np.ndarray:
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)


def get_red_mask(
    hsv: np.ndarray,
    h1_hi: int = 15,
    h2_lo: int = 160,
    s_lo: int = 60,
    v_lo: int = 60,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """빨간색 HSV 마스크 (작은 스티커 보존)."""
    lo1 = np.array([0, s_lo, v_lo], dtype=np.uint8)
    hi1 = np.array([h1_hi, 255, 255], dtype=np.uint8)
    lo2 = np.array([h2_lo, s_lo, v_lo], dtype=np.uint8)
    hi2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lo1, hi1)
    m2 = cv2.inRange(hsv, lo2, hi2)
    mask = cv2.bitwise_or(m1, m2)

    # 원형 판 내부 영역으로만 제한 (파란색 마스크가 아닌 원형 판 영역 ROI)
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    # 아주 작은 스티커가 지워지지 않도록 작은 팽창(Dilation) 적용
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.dilate(mask, kern, iterations=1)


def find_largest_circle(mask: np.ndarray, min_area: float = 300.0) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
    """가장 큰 파란색 원형 판의 중심 (cx, cy), 반지름 R, contour 반환."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    (x, y), radius = cv2.minEnclosingCircle(c)
    return (int(round(x)), int(round(y))), float(radius), c


def find_marker_centroid(
    mask: np.ndarray,
    center: Tuple[int, int],
    radius: float,
    min_area: float = 2.0,
    max_area: float = 4000.0,
) -> Optional[Tuple[int, int]]:
    """원형 판 내부에서 빨간색 스티커의 무게중심 (mx, my) 반환."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = center
    best_pt = None
    max_score = -1.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1e-4:
            # 픽셀이 1~2개로 매우 작을 경우 Bounding Box 중심 사용
            bx, by, bw, bh = cv2.boundingRect(c)
            mx = int(bx + bw // 2)
            my = int(by + bh // 2)
        else:
            mx = int(round(M["m10"] / M["m00"]))
            my = int(round(M["m01"] / M["m00"]))

        dist_from_center = np.hypot(mx - cx, my - cy)
        # 스티커는 중심에서 바깥쪽(0.1*R ~ 1.15*R)에 위치
        if 0.1 * radius <= dist_from_center <= 1.15 * radius:
            score = area
            if score > max_score:
                max_score = score
                best_pt = (mx, my)

    return best_pt


def align_and_crop(
    bgr: np.ndarray,
    center: Tuple[int, int],
    marker: Tuple[int, int],
    radius: float,
    out_size: int = 256,
) -> Tuple[np.ndarray, float]:
    """기준점 스티커가 12시 방향(-90°)을 가리키도록 회전 및 out_size 정규화 크롭."""
    cx, cy = center
    mx, my = marker

    # 현재 스티커 각도 (라디안 -> 디그리)
    current_deg = np.degrees(np.arctan2(my - cy, mx - cx))
    # 12시 방향(-90도)으로 만들기 위한 회전 각도
    rot_deg = current_deg - (-90.0)

    # 스케일 조정 (원형 판이 out_size의 85% 크기가 되도록)
    scale = (out_size * 0.42) / max(radius, 10.0)

    # 2D 아핀 회전 행렬
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale)
    # 회전 후 새 이미지의 중앙(out_size//2, out_size//2)으로 평행이동
    M[0, 2] += (out_size / 2.0) - cx
    M[1, 2] += (out_size / 2.0) - cy

    aligned = cv2.warpAffine(
        bgr,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, rot_deg


def main():
    parser = argparse.ArgumentParser(description="HSV Mask & Auto-Alignment Tester")
    parser.add_argument("--cam-id", type=int, default=0, help="USB Camera ID (default 0)")
    parser.add_argument("--use-webcam", action="store_true", help="Force USB webcam instead of HP60C shm")
    args = parser.parse_args()

    # 1. 설정 불러오기
    cfg_path = _HERE / "depth_cfg.json"
    blue_lo = DEFAULT_BLUE_LO
    blue_hi = DEFAULT_BLUE_HI
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            di = cfg.get("depth_inspect", cfg)
            hb = di.get("hsv_blue", di.get("hsv", {}).get("blue", []))
            if len(hb) >= 2:
                blue_lo = tuple(hb[0])
                blue_hi = tuple(hb[1])
        except Exception:
            pass

    red_lo = (0, 100, 100)
    red_hi = (180, 255, 255)

    # 2. 카메라 연결
    shm_reader: Optional[ShmDepthReader] = None
    cap: Optional[cv2.VideoCapture] = None

    if not args.use_webcam:
        shm_reader = ShmDepthReader()
        if shm_reader.is_available():
            print("[INFO] HP60C 공유메모리(/dev/shm/hp60c_frames)에서 RGB 영상을 읽습니다.")
        else:
            print("[INFO] HP60C 공유메모리가 없어 USB 카메라(ID 0)를 엽니다.")
            cap = cv2.VideoCapture(args.cam_id)
    else:
        cap = cv2.VideoCapture(args.cam_id)

    # 3. 창 및 트랙바 생성
    setup_trackbars(blue_lo, blue_hi, red_s_lo=60, red_v_lo=60)

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, 640, 480)

    cv2.namedWindow(WIN_ALIGNED, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_ALIGNED, 300, 300)

    print("\n=======================================================")
    print(" [HSV 마스크 & 자동 회전 정렬 검증 도구] ")
    print(" - 파란색 원형 판과 빨간색 스티커를 카메라에 비춰주세요.")
    print(" - 단축키: 's' = 현재 HSV 저장 | 'q' or ESC = 종료")
    print("=======================================================\n")

    while True:
        frame = None
        if shm_reader is not None and shm_reader.is_available():
            rgb, _, _ = shm_reader.read()
            if rgb is not None:
                frame = rgb.copy()
        elif cap is not None and cap.isOpened():
            ret, f = cap.read()
            if ret:
                frame = f

        if frame is None:
            # 대기 화면
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Waiting for camera stream...",
                (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow(WIN_MAIN, blank)
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
            continue

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 트랙바 값 읽기
        b_h_lo = cv2.getTrackbarPos("Blue H Lo", WIN_MASKS)
        b_h_hi = cv2.getTrackbarPos("Blue H Hi", WIN_MASKS)
        b_s_lo = cv2.getTrackbarPos("Blue S Lo", WIN_MASKS)
        b_s_hi = cv2.getTrackbarPos("Blue S Hi", WIN_MASKS)
        b_v_lo = cv2.getTrackbarPos("Blue V Lo", WIN_MASKS)
        b_v_hi = cv2.getTrackbarPos("Blue V Hi", WIN_MASKS)
        r_h1_hi = cv2.getTrackbarPos("Red H1 Hi", WIN_MASKS)
        r_h2_lo = cv2.getTrackbarPos("Red H2 Lo", WIN_MASKS)
        r_s_lo = cv2.getTrackbarPos("Red S Lo", WIN_MASKS)
        r_v_lo = cv2.getTrackbarPos("Red V Lo", WIN_MASKS)
        r_min_area = cv2.getTrackbarPos("Red Min Area", WIN_MASKS)

        cur_blue_lo = (b_h_lo, b_s_lo, b_v_lo)
        cur_blue_hi = (b_h_hi, b_s_hi, b_v_hi)

        # 1. 파란색 원형 판 마스크
        blue_mask = get_blue_mask(hsv, cur_blue_lo, cur_blue_hi)
        plate_info = find_largest_circle(blue_mask)

        # 원형 판 영역 ROI 마스크 생성 (파란색 마스크가 아닌 원형 판 영역 전체)
        roi_mask = None
        if plate_info is not None:
            (cx, cy), radius, _ = plate_info
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(roi_mask, (cx, cy), int(radius * 1.1), 255, -1)

        # 2. 빨간색 스티커 마스크
        red_mask = get_red_mask(
            hsv,
            h1_hi=r_h1_hi,
            h2_lo=r_h2_lo,
            s_lo=r_s_lo,
            v_lo=r_v_lo,
            roi_mask=roi_mask,
        )

        overlay = frame.copy()
        aligned_roi = np.zeros((256, 256, 3), dtype=np.uint8)

        if plate_info is not None:
            (cx, cy), radius, cnt = plate_info
            # 파란색 원형 판 그리기
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), int(radius), (255, 255, 0), 1)
            cv2.drawMarker(overlay, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

            # 스티커 무게중심 찾기
            marker = find_marker_centroid(red_mask, (cx, cy), radius, min_area=float(r_min_area))

            if marker is not None:
                mx, my = marker
                # 스티커 위치 표시 (Red circle + crosshair)
                cv2.circle(overlay, (mx, my), 8, (0, 0, 255), -1)
                cv2.circle(overlay, (mx, my), 12, (255, 255, 255), 2)
                # 방향 벡터 (Center -> Marker) 화살표
                cv2.arrowedLine(overlay, (cx, cy), (mx, my), (0, 0, 255), 3, tipLength=0.2)

                # 12시 방향 자동 회전 정렬
                aligned_roi, rot_deg = align_and_crop(frame, (cx, cy), (mx, my), radius, out_size=256)

                # 정렬 화면에 12시 기준 가이드 라인 표시
                cv2.drawMarker(aligned_roi, (128, 128), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.circle(aligned_roi, (128, 128), int(256 * 0.42), (0, 255, 0), 1)
                # 12시 방향 스티커 지점 마커
                cv2.circle(aligned_roi, (128, int(128 - 256 * 0.35)), 6, (0, 0, 255), -1)

                cv2.putText(
                    overlay,
                    f"STATUS: ALIGNED (Rot: {rot_deg:+.1f} deg)",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    overlay,
                    "STATUS: Plate OK, Red Sticker NOT Found",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    aligned_roi,
                    "Sticker Missing",
                    (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    1,
                )
        else:
            cv2.putText(
                overlay,
                "STATUS: Waiting for Blue Plate...",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 180, 180),
                2,
            )
            cv2.putText(
                aligned_roi,
                "No Plate",
                (75, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
            )

        # 마스크 합성 뷰 (좌: 파란색 마스크, 우: 빨간색 스티커 마스크)
        b_mask_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
        r_mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(b_mask_bgr, "BLUE PLATE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(r_mask_bgr, "RED STICKER MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        masks_combo = np.hstack([cv2.resize(b_mask_bgr, (400, 300)), cv2.resize(r_mask_bgr, (400, 300))])

        # 화면 출력
        cv2.imshow(WIN_MAIN, overlay)
        cv2.imshow(WIN_MASKS, masks_combo)
        cv2.imshow(WIN_ALIGNED, aligned_roi)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            # depth_cfg.json에 저장
            cfg_data = {}
            if cfg_path.is_file():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            di = cfg_data.get("depth_inspect", {})
            di["hsv_blue"] = [list(cur_blue_lo), list(cur_blue_hi)]
            cfg_data["depth_inspect"] = di
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2, ensure_ascii=False)
            print(f"\n[SAVED] Blue HSV 저장 완료: LO={cur_blue_lo}, HI={cur_blue_hi} -> {cfg_path}")

    if cap is not None:
        cap.release()
    if shm_reader is not None:
        shm_reader.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
