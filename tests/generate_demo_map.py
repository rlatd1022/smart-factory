#!/usr/bin/env python3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cv2
from iotdemo.vision.wafer_spatial_analyzer import WaferSpatialAnalyzer

def main():
    analyzer = WaferSpatialAnalyzer(wafer_radius_mm=45.0, grid_size=256)

    # 1. Normal samples
    for i in range(15):
        analyzer.add_inspection_event(i + 1, False, [])

    # 2. Defect cluster in Q1 (NE) Edge - Scratches
    for i in range(6):
        analyzer.add_inspection_event(
            16 + i,
            True,
            [
                {"r_mm": 37.0 + i * 0.5, "theta_deg": 35.0 + i * 5.0, "defect_type": "SCRATCH", "area_px": 30.0, "cx_px": 160, "cy_px": 160},
                {"r_mm": 40.0, "theta_deg": 50.0, "defect_type": "MARKER_X", "area_px": 50.0, "cx_px": 170, "cy_px": 170},
            ],
        )

    # 3. Foreign spot in Center
    analyzer.add_inspection_event(
        22,
        True,
        [{"r_mm": 10.0, "theta_deg": 180.0, "defect_type": "FOREIGN_SPOT", "area_px": 18.0, "cx_px": 120, "cy_px": 130}],
    )

    img = analyzer.generate_wafer_map_image((300, 300))
    out_path = "/home/intel/.gemini/antigravity/brain/45371433-5d14-43ec-9fc9-5db5d98b9983/wafer_spatial_map_demo.png"
    cv2.imwrite(out_path, img)
    print(f"Saved demo map image to {out_path}")

    summary = analyzer.get_summary()
    print("Summary:", summary)

if __name__ == "__main__":
    main()
