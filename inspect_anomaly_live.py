#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_anomaly_live.py — 실시간 웹캠/HP60C 기반 12시 자동 정렬 및 PatchCore 다중 결함 이상치 검사 뷰어.

사용법:
    cd /home/intel/smart_factory_edu
    python inspect_anomaly_live.py --use-webcam --cam-id 0
"""

from __future__ import annotations

import argparse
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

from iotdemo.deeplearning.anomaly_detector import PatchCoreDetector
from iotdemo.depth.depth_window_inspector import ShmDepthReader
from iotdemo.vision.wafer_aligner import WaferAligner

WIN_LIVE = "Wafer Live Vision Inspection"
WIN_HEATMAP = "PatchCore Multi-Defect Heatmap (256x256)"


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
    return {"blue_lo": blue_lo, "blue_hi": blue_hi}


def main():
    parser = argparse.ArgumentParser(description="Live Real-time PatchCore Multi-Defect Inspector")
    parser.add_argument("--weights", type=str, default="weights/patchcore_resnet18.pkl", help="Trained weights file")
    parser.add_argument("--cam-id", type=int, default=0, help="USB Camera ID (default 0)")
    parser.add_argument("--use-webcam", action="store_true", help="Force USB webcam instead of HP60C shm")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold override")
    args = parser.parse_args()

    weights_path = _HERE / args.weights
    if not weights_path.is_file():
        print(f"[ERROR] 모델 가중치 파일이 없습니다: {weights_path}")
        print("먼저 `python train_anomaly.py`를 실행해 주세요.")
        sys.exit(1)

    print(f"\n[INFO] PatchCore 다중 결함 이상치 모델 로드 중: {weights_path}")
    detector = PatchCoreDetector()
    detector.load(str(weights_path))

    if args.threshold is not None:
        detector.threshold = args.threshold
        print(f"[INFO] 임계값 수동 지정: {detector.threshold:.4f}")

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
            print("[INFO] HP60C 공유메모리에서 영상을 읽습니다.")
        else:
            print("[INFO] HP60C 공유메모리가 없어 USB 카메라를 엽니다.")
            cap = cv2.VideoCapture(args.cam_id)
    else:
        cap = cv2.VideoCapture(args.cam_id)

    cv2.namedWindow(WIN_LIVE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_LIVE, 720, 540)
    cv2.namedWindow(WIN_HEATMAP, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_HEATMAP, 350, 350)

    print("\n=======================================================")
    print(" [실시간 PatchCore 다중 결함 동시 검사 뷰어 가동] ")
    print(f" - 판정 기준 임계값: {detector.threshold:.4f}")
    print(" - 여러 군데의 결함을 동시에 개별 추적 및 극좌표 산출")
    print(" - 단축키: [+/-]: 임계값 미세조절 | [q]: 종료")
    print("=======================================================\n")

    latest_heatmap_view = np.zeros((256, 256, 3), dtype=np.uint8)

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
            cv2.imshow(WIN_LIVE, blank)
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
            continue

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. 자동 회전 정렬
        t0 = time.time()
        result = aligner.process(frame)

        if result is not None:
            # 2. PatchCore 다중 결함 이상치 검사 수행
            anomaly_res = detector.predict(result.aligned_img)
            infer_ms = (time.time() - t0) * 1000.0

            cx, cy = result.plate_info.center
            radius = result.plate_info.radius
            mx, my = result.marker_info.centroid

            # 판 및 스티커 표시
            cv2.drawContours(overlay, [result.plate_info.contour], -1, (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), int(radius), (255, 255, 0), 1)
            cv2.circle(overlay, (mx, my), 7, (0, 0, 255), -1)
            cv2.arrowedLine(overlay, (cx, cy), (mx, my), (0, 0, 255), 2, tipLength=0.2)

            # 판정 상태 표시
            if anomaly_res.is_defect:
                status_color = (0, 0, 255)  # Red
                status_label = f"DEFECT (NG) | Count: {anomaly_res.num_defects} | Score: {anomaly_res.anomaly_score:.3f}"
            else:
                status_color = (0, 255, 0)  # Green
                status_label = f"NORMAL (OK) | Score: {anomaly_res.anomaly_score:.3f}"

            cv2.putText(overlay, status_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
            cv2.putText(
                overlay,
                f"Thresh: {detector.threshold:.3f} | Infer: {infer_ms:.1f}ms | Dispersion: {anomaly_res.dispersion_mm:.1f}mm",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            # 검출된 모든 결함의 원본 영상 매핑 및 극좌표 리스트 표시
            if anomaly_res.is_defect and anomaly_res.defects:
                for d in anomaly_res.defects:
                    u, v = d.centroid_pix
                    orig_dx, orig_dy = result.aligned_to_original(u, v)
                    odx_i, ody_i = int(round(orig_dx)), int(round(orig_dy))

                    # 원본 화면 상의 결함 마킹 및 번호 배지
                    cv2.drawMarker(overlay, (odx_i, ody_i), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
                    cv2.circle(overlay, (odx_i, ody_i), 12, (0, 0, 255), 2)
                    cv2.putText(
                        overlay,
                        f"#{d.defect_id}",
                        (odx_i + 14, ody_i + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                # 결함 극좌표 목록 텍스트 오버레이 (최대 4개 표시)
                text_y = 100
                for d in anomaly_res.defects[:4]:
                    r_mm, theta_deg = d.polar_coords
                    cv2.putText(
                        overlay,
                        f"#{d.defect_id}: r={r_mm:.1f}mm, {theta_deg:.0f}deg ({d.clock_pos})",
                        (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        1,
                    )
                    text_y += 25

            # 히트맵 뷰 업데이트
            latest_heatmap_view = anomaly_res.overlay_bgr.copy()
            # 12시 스티커 마크 표시
            cv2.circle(latest_heatmap_view, (128, int(128 - 256 * 0.35)), 6, (0, 0, 255), -1)
            cv2.putText(
                latest_heatmap_view,
                "12 o'clock",
                (95, int(128 - 256 * 0.35) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

            # 히트맵 상단 판정 텍스트
            header_txt = f"NG ({anomaly_res.num_defects} defects)" if anomaly_res.is_defect else "OK (NORMAL)"
            cv2.putText(
                latest_heatmap_view,
                header_txt,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
            )
        else:
            cv2.putText(overlay, "Searching for Wafer...", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(WIN_LIVE, overlay)
        cv2.imshow(WIN_HEATMAP, latest_heatmap_view)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("+") or key == ord("="):
            detector.threshold += 0.02
            print(f"[INFO] 임계값 증가 -> {detector.threshold:.4f}")
        elif key == ord("-") or key == ord("_"):
            detector.threshold = max(detector.threshold - 0.02, 0.05)
            print(f"[INFO] 임계값 감소 -> {detector.threshold:.4f}")

    if cap is not None:
        cap.release()
    if shm_reader is not None:
        shm_reader.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
