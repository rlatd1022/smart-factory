#!/usr/bin/env python3
"""
Unit test for WaferSpatialAnalyzer.
Verifies:
1. Spatial coordinate mapping and Gaussian density heatmap generation
2. Sector/Quadrant defect occurrence probability calculation
3. Hotspot diagnosis alert generation
4. Virtual wafer map image rendering
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cv2
import numpy as np
from iotdemo.vision.wafer_spatial_analyzer import WaferSpatialAnalyzer


def test_spatial_analyzer_hotspot_simulation():
    analyzer = WaferSpatialAnalyzer(wafer_radius_mm=45.0, grid_size=256)

    # 1. Normal (PASS) wafers 10 samples
    for i in range(10):
        analyzer.add_inspection_event(object_id=i + 1, is_defect=False, defects_info=[])

    summary = analyzer.get_summary()
    assert summary.total_wafers == 10
    assert summary.defect_wafers == 0
    assert summary.overall_defect_rate == 0.0
    assert "정상" in summary.tendency_diagnosis
    print("✅ Pass 10 samples verified: Clean state confirmed")

    # 2. Simulate Q1 (NE: 0~90 deg) Outer Scratch Defects (5 defective wafers with scratches in Q1, r=38mm, theta=45 deg)
    for i in range(5):
        defects = [
            {"r_mm": 38.0, "theta_deg": 45.0, "defect_type": "SCRATCH", "area_px": 25.0, "cx_px": 150, "cy_px": 150}
        ]
        analyzer.add_inspection_event(object_id=10 + i + 1, is_defect=True, defects_info=defects)

    # 3. Simulate 1 Q3 (SW: 180~270 deg) Foreign Spot
    analyzer.add_inspection_event(
        object_id=16,
        is_defect=True,
        defects_info=[{"r_mm": 20.0, "theta_deg": 225.0, "defect_type": "FOREIGN_SPOT", "area_px": 15.0, "cx_px": 140, "cy_px": 170}],
    )

    # 4. Simulate 4 more normal wafers
    for i in range(4):
        analyzer.add_inspection_event(object_id=17 + i, is_defect=False, defects_info=[])

    summary = analyzer.get_summary()
    assert summary.total_wafers == 20
    assert summary.defect_wafers == 6
    assert summary.total_defects == 6
    assert abs(summary.overall_defect_rate - 30.0) < 1e-3

    # Q1 Probability: 5 wafers out of 20 = 25.0%
    assert abs(summary.quadrant_prob["Q1_NE"] - 25.0) < 1e-3
    # Q3 Probability: 1 wafer out of 20 = 5.0%
    assert abs(summary.quadrant_prob["Q3_SW"] - 5.0) < 1e-3
    # Q2 & Q4: 0.0%
    assert summary.quadrant_prob["Q2_SE"] == 0.0
    assert summary.quadrant_prob["Q4_NW"] == 0.0

    # Radial: EDGE should be highest
    assert summary.radial_counts["EDGE"] == 5
    assert summary.radial_counts["MID"] == 1
    assert summary.radial_counts["CENTER"] == 0

    # Diagnosis should detect Q1 hotspot and Edge clustering
    assert summary.is_hotspot_detected is True
    assert "1사분면(NE)" in summary.tendency_diagnosis
    assert "외곽 엣지" in summary.tendency_diagnosis
    print(f"✅ Hotspot diagnosis verified: {summary.tendency_diagnosis}")

    # 5. Image rendering test
    img = analyzer.generate_wafer_map_image(target_size=(220, 220))
    assert isinstance(img, np.ndarray)
    assert img.shape == (220, 220, 3)
    assert np.max(img) > 0
    print("✅ Virtual Wafer Map image (220x220x3) rendered successfully")

    # 6. Reset test
    analyzer.reset()
    reset_summary = analyzer.get_summary()
    assert reset_summary.total_wafers == 0
    assert reset_summary.defect_wafers == 0
    assert reset_summary.total_defects == 0
    assert np.max(analyzer.density_map) == 0.0
    print("✅ Analyzer reset verified")

    print("\n🎉 ALL WAFER SPATIAL ANALYZER TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_spatial_analyzer_hotspot_simulation()
