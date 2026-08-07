#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_anomaly.py — 정상 샘플 데이터셋으로 PatchCore 비지도 이상치 탐지 모델 학습."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# smart_factory_edu root
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from iotdemo.deeplearning.anomaly_detector import PatchCoreDetector


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore Anomaly Detection Model on Normal Samples")
    parser.add_argument("--data-dir", type=str, default="dataset/train/good", help="Path to normal samples directory")
    parser.add_argument("--weights", type=str, default="weights/patchcore_resnet18.pkl", help="Output model weights file")
    parser.add_argument("--subsample", type=float, default=0.1, help="Coreset subsample ratio (default 0.1)")
    parser.add_argument("--radius-mm", type=float, default=40.0, help="Physical wafer radius in mm (default 40.0)")
    args = parser.parse_args()

    data_dir = _HERE / args.data_dir
    weights_path = _HERE / args.weights

    if not data_dir.is_dir():
        print(f"[ERROR] 데이터 폴더를 찾을 수 없습니다: {data_dir}")
        print("먼저 `python collect_samples.py`를 실행하여 정상 샘플을 수집해 주세요.")
        sys.exit(1)

    img_paths = sorted(
        list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg"))
    )

    if not img_paths:
        print(f"[ERROR] {data_dir} 에 학습용 이미지가 없습니다.")
        print("먼저 `python collect_samples.py`를 실행하여 정상 샘플을 수집해 주세요.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" [PatchCore 비지도 이상치 탐지 모델 학습] ")
    print(f" - 데이터셋 경로: {data_dir} ({len(img_paths)}장)")
    print(f" - 가중치 저장 경로: {weights_path}")
    print(f"=======================================================\n")

    images = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)

    if len(images) < 3:
        print(f"[WARNING] 이미지가 {len(images)}장으로 너무 적습니다. 최소 10~30장 수집을 권장합니다.")

    detector = PatchCoreDetector(
        wafer_radius_mm=args.radius_mm,
        subsample_ratio=args.subsample,
        mask_sticker_region=True,
    )

    # 1. 학습 수행 (Memory Bank 생성 및 임계값 산출)
    threshold = detector.fit(images)

    # 2. 모델 저장
    detector.save(str(weights_path))

    # 3. 자체 검증 테스트 (정상 샘플 vs 가상 불량 샘플)
    print("\n--- [학습 모델 자체 검증 테스트] ---")
    sample_img = images[0]
    res_normal = detector.predict(sample_img)
    print(f"[1. 정상 샘플 테스트] 이상치 점수: {res_normal.anomaly_score:.4f} (임계값: {threshold:.4f}) -> 판정: {'[불량 NG]' if res_normal.is_defect else '[정상 OK]'}")

    # 가상 스크래치 결함 추가 테스트
    defect_sim_img = sample_img.copy()
    cv2.line(defect_sim_img, (140, 110), (180, 140), (0, 0, 0), 3)  # 검은색 스크래치
    cv2.circle(defect_sim_img, (100, 160), 6, (255, 255, 255), -1)  # 흰색 이물
    res_defect = detector.predict(defect_sim_img)
    print(f"[2. 가상 불량 테스트] 이상치 점수: {res_defect.anomaly_score:.4f} (임계값: {threshold:.4f}) -> 판정: {'[불량 NG (정상 검출!)]' if res_defect.is_defect else '[정상 OK]'}")
    if res_defect.polar_coords:
        r_mm, theta_deg = res_defect.polar_coords
        print(f"    └─ 검출된 결함 극좌표: 반경 {r_mm:.1f}mm, 12시기준 각도 {theta_deg:.1f}°")

    print("\n[성공] 학습이 완벽하게 완료되었습니다! 스마트팩토리 공정에서 실시간 불량 검사를 수행할 수 있습니다.\n")


if __name__ == "__main__":
    main()
