#!/usr/bin/env python3
"""Display and measure the HP60C depth stream."""

from __future__ import annotations

from argparse import ArgumentParser

import cv2
import numpy as np

from iotdemo.depth.hp60c import HP60CCamera, depth_at, discover_depth_devices


WINDOW = "HP60C depth (click to measure, q to quit)"


def colorize(
    depth: np.ndarray, scale: float, min_m: float, max_m: float
) -> np.ndarray:
    depth_m = depth.astype(np.float32) * scale
    valid = (depth_m >= min_m) & (depth_m <= max_m)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (max_m - depth_m[valid]) * 255.0 / (max_m - min_m), 0, 255
    ).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def main() -> int:
    parser = ArgumentParser(description="Measure distance with a YDLIDAR HP60C")
    parser.add_argument(
        "-c",
        "--camera",
        help="Depth node, e.g. /dev/video2 (auto-detected by default)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List RAW16 depth nodes and exit"
    )
    parser.add_argument(
        "--scale", type=float, default=0.001, help="Meters per RAW16 unit"
    )
    parser.add_argument("--min", dest="min_m", type=float, default=0.2)
    parser.add_argument("--max", dest="max_m", type=float, default=4.0)
    parser.add_argument(
        "--radius", type=int, default=2, help="Measurement median radius"
    )
    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be greater than zero")
    if args.min_m < 0 or args.max_m <= args.min_m:
        parser.error("--max must be greater than --min, and --min cannot be negative")
    if args.radius < 0:
        parser.error("--radius cannot be negative")

    devices = discover_depth_devices()
    if args.list:
        if devices:
            print("\n".join(devices))
        else:
            print("No V4L2 RAW16/Y16 depth nodes found")
        return 0

    selected: list[int] = []

    def select(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [x, y]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, select)

    try:
        with HP60CCamera(args.camera) as camera:
            print(f"Using HP60C depth device: {camera.device}")
            while True:
                depth = camera.read()
                image = colorize(depth, args.scale, args.min_m, args.max_m)
                x, y = selected if selected else [
                    depth.shape[1] // 2,
                    depth.shape[0] // 2,
                ]
                distance = depth_at(
                    depth,
                    x,
                    y,
                    radius=args.radius,
                    scale=args.scale,
                    min_m=args.min_m,
                    max_m=args.max_m,
                )
                label = "invalid" if distance is None else f"{distance:.3f} m"
                cv2.drawMarker(
                    image, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2
                )
                cv2.putText(
                    image,
                    f"({x}, {y}) {label}",
                    (max(5, x - 80), max(25, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW, image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
