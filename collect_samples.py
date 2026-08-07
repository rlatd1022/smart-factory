#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_samples.py — 컨베이어 통과 시 12시 방향 자동 정렬 정상(양품) 데이터 자동 수집기.

사용법:
    cd /home/intel/smart_factory_edu
    python collect_samples.py --target 50

기능:
  - 컨베이어 벨트를 통과하는 파란색 원형 판 + 빨간색 스티커를 자동 감지
  - 화면 중앙 검사 구역(Center Gate)을 지날 때 12시 방향으로 자동 회전 정렬된 256x256 이미지 저장
  - 중복 저장 방지(Debounce & Transit Gate)
  - 목표 수량(기본 50장) 달성 시 자동 완료
  - 단축키:
      [Space]: 수동 즉시 캡처
      [r]: 수집 개수 리셋
      [q] 또는 [ESC]: 종료
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# smart_factory_edu root
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from iotdemo.depth.depth_window_inspector import ShmDepthReader
from iotdemo.vision.wafer_aligner import WaferAligner

WIN_MAIN = "Conveyor Auto Collector - Camera Stream"
WIN_PREVIEW = "Latest Aligned Normal Sample (256x256)"


def load_config(use_webcam: bool = False) -> dict:
    blue_lo = (102, 175, 118)
    blue_hi = (179, 255, 255)

    if use_webcam:
        color_cfg = _HERE / "color.cfg"
        if color_cfg.is_file():
            try:
                import configparser

                cp = configparser.ConfigParser()
                cp.read(color_cfg)
                if cp.has_section("default") and cp.has_option("default", "blue"):
                    val = eval(cp.get("default", "blue"))
                    if len(val) >= 6:
                        blue_lo = (int(val[0]), int(val[1]), int(val[2]))
                        blue_hi = (int(val[3]), int(val[4]), int(val[5]))
            except Exception:
                pass
    else:
        cfg_path = _HERE / "depth_cfg.json"
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
    return {"blue_lo": blue_lo, "blue_hi": blue_hi}


def main():
    parser = argparse.ArgumentParser(description="Conveyor Auto Normal Sample Collector")
    parser.add_argument("--target", type=int, default=100, help="Target number of normal samples (default 100)")
    parser.add_argument("--output-dir", type=str, default="dataset/train/good", help="Output directory")
    parser.add_argument("--cam-id", type=int, default=0, help="USB Camera ID (default 0)")
    parser.add_argument("--use-webcam", action="store_true", help="Force USB webcam instead of HP60C shm")
    parser.add_argument("--clear", action="store_true", help="Clear existing dataset folder before collecting")
    parser.add_argument("--interval", type=float, default=0.6, help="Min seconds between captures (default 0.6s)")
    parser.add_argument("--gate-ratio", type=float, default=0.4, help="Central inspection gate size ratio (default 0.4)")
    args = parser.parse_args()

    out_dir = _HERE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for f in out_dir.glob("good_*.png"):
            try:
                f.unlink()
            except Exception:
                pass
        print(f"[INFO] 기존 데이터셋({out_dir})을 모두 초기화했습니다.")

    # 기존 저장된 파일 개수 파악
    existing_files = sorted(list(out_dir.glob("good_*.png")))
    count = len(existing_files)
    print(f"\n[INFO] 저장 경로: {out_dir}")
    print(f"[INFO] 기존 수집된 파일: {count}장 / 목표 수량: {args.target}장")

    cfg = load_config(use_webcam=args.use_webcam)
    aligner = WaferAligner(
        blue_lo=cfg["blue_lo"],
        blue_hi=cfg["blue_hi"],
        red_h1_hi=15,
        red_h2_lo=160,
        red_s_lo=60,
        red_v_lo=60,
        red_min_area=2.0,
        plate_min_area=300.0,
        out_size=256,
    )

    # 카메라 연결
    shm_reader: Optional[ShmDepthReader] = None
    cap: Optional[cv2.VideoCapture] = None

    if not args.use_webcam:
        shm_reader = ShmDepthReader()
        if shm_reader.is_available():
            print("[INFO] HP60C 공유메모리(/dev/shm/hp60c_frames)에서 영상을 읽습니다.")
        else:
            print("[INFO] HP60C 공유메모리가 없어 USB 카메라(ID 0)를 엽니다.")
            cap = cv2.VideoCapture(args.cam_id)
    else:
        cap = cv2.VideoCapture(args.cam_id)

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, 720, 540)
    cv2.namedWindow(WIN_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PREVIEW, 320, 320)

    last_save_time = 0.0
    was_inside_gate = False
    flash_frames = 0
    latest_saved_img = np.zeros((256, 256, 3), dtype=np.uint8)

    print("\n=======================================================")
    print(" [컨베이어 자동 정상 데이터 수집기 가동] ")
    print(f" - 정상 샘플을 컨베이어에 흘려보내면 {args.target}장이 자동 수집됩니다.")
    print(" - [Space]: 수동 캡처 | [r]: 리셋 | [q]: 종료")
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
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WIN_MAIN, blank)
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
            continue

        h, w = frame.shape[:2]
        now = time.time()

        # 중앙 검사 구역 (Gate Box)
        gw = int(w * args.gate_ratio)
        gh = int(h * args.gate_ratio)
        gx1, gy1 = (w - gw) // 2, (h - gh) // 2
        gx2, gy2 = gx1 + gw, gy1 + gh

        # 정렬 모듈 처리
        result = aligner.process(frame)
        overlay = frame.copy()

        # 중앙 게이트 박스 표시
        gate_color = (0, 255, 255)
        is_inside_gate = False
        save_trigger = False

        if result is not None:
            cx, cy = result.plate_info.center
            radius = result.plate_info.radius
            mx, my = result.marker_info.centroid

            # 판 및 스티커 표시
            cv2.drawContours(overlay, [result.plate_info.contour], -1, (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), int(radius), (255, 255, 0), 1)
            cv2.circle(overlay, (mx, my), 7, (0, 0, 255), -1)
            cv2.arrowedLine(overlay, (cx, cy), (mx, my), (0, 0, 255), 2, tipLength=0.2)

            # 게이트 진입 여부 판별
            if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                is_inside_gate = True
                gate_color = (0, 255, 0)

                # 자동 캡처 조건: 게이트에 진입한 순간(Edge-trigger) + 쿨다운 시간 충족 + 미완료
                if (not was_inside_gate or (now - last_save_time > args.interval)) and count < args.target:
                    save_trigger = True
            else:
                gate_color = (0, 255, 255)

            was_inside_gate = is_inside_gate
        else:
            was_inside_gate = False

        # 이미지 저장 수행
        if save_trigger and result is not None:
            count += 1
            filename = out_dir / f"good_{count:04d}.png"
            cv2.imwrite(str(filename), result.aligned_img)
            last_save_time = now
            flash_frames = 5
            latest_saved_img = result.aligned_img.copy()
            print(f"[CAPTURED] #{count}/{args.target} 저장 완료 -> {filename.name} (회전각: {result.rot_deg:+.1f}°)")

        # 캡처 플래시 효과
        if flash_frames > 0:
            flash_overlay = np.full_like(overlay, 255)
            overlay = cv2.addWeighted(overlay, 0.7, flash_overlay, 0.3, 0)
            flash_frames -= 1

        # 게이트 사각형 그리기
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), gate_color, 2)
        cv2.putText(overlay, "INSPECTION GATE", (gx1 + 5, gy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gate_color, 1)

        # 프로그레스 바 UI
        progress = min(float(count) / max(args.target, 1), 1.0)
        bar_w = int((w - 40) * progress)
        cv2.rectangle(overlay, (20, h - 35), (w - 20, h - 15), (50, 50, 50), -1)
        cv2.rectangle(overlay, (20, h - 35), (20 + bar_w, h - 15), (0, 220, 0), -1)
        cv2.rectangle(overlay, (20, h - 35), (w - 20, h - 15), (200, 200, 200), 1)

        status_text = f"Collected: {count} / {args.target} ({progress * 100:.0f}%)"
        if count >= args.target:
            status_text += " [COMPLETED! Press 'q' to train]"
            cv2.putText(overlay, status_text, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, status_text, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 미리보기 창 업데이트
        preview_disp = latest_saved_img.copy()
        if count > 0:
            cv2.putText(
                preview_disp,
                f"Saved #{count}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.circle(preview_disp, (128, int(128 - 256 * 0.35)), 5, (0, 0, 255), -1)

        cv2.imshow(WIN_MAIN, overlay)
        cv2.imshow(WIN_PREVIEW, preview_disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            # 수동 즉시 캡처
            if result is not None:
                count += 1
                filename = out_dir / f"good_{count:04d}.png"
                cv2.imwrite(str(filename), result.aligned_img)
                last_save_time = now
                flash_frames = 5
                latest_saved_img = result.aligned_img.copy()
                print(f"[MANUAL CAPTURE] #{count}/{args.target} 저장 완료 -> {filename.name}")
        elif key == ord("r"):
            count = 0
            print("[INFO] 수집 개수를 리셋했습니다.")

    if cap is not None:
        cap.release()
    if shm_reader is not None:
        shm_reader.close()
    cv2.destroyAllWindows()

    print(f"\n[완료] 총 {count}장의 정렬된 정상 데이터가 {out_dir} 에 저장되었습니다.")
    if count >= 10:
        print("다음 단계: python train_anomaly.py 를 실행하여 비지도 이상치 모델을 학습시키세요!\n")


if __name__ == "__main__":
    main()
