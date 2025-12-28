#!/usr/bin/env python3
"""
Simple script to create and manage color.cfg for color detection
Supports interactive HSV color space filtering with config saving on exit
"""

import sys
import cv2
import numpy as np
from functools import partial
from configparser import ConfigParser


def save_color_config(filename, colors_dict):
    """Save color configuration to INI format file

    Args:
        filename: Output file path (e.g., 'color.cfg')
        colors_dict: Dictionary of color_name -> (h_min, s_min, v_min, h_max, s_max, v_max, dilate) tuples
    """
    config = ConfigParser()
    config["default"] = {}

    for name, values in colors_dict.items():
        # Format as tuple string for the preset loader
        config["default"][name] = str(values)

    with open(filename, "w") as f:
        config.write(f)

    print(f"Saved {filename}:")
    for name, values in colors_dict.items():
        print(
            f"  {name}: H({values[0]}-{values[3]}) S({values[1]}-{values[4]}) V({values[2]}-{values[5]})"
        )


def update_color_value(x, color, is_min):
    global h_min, s_min, v_min, h_max, s_max, v_max
    match color:
        case "H":
            if is_min:
                h_min = x
            else:
                h_max = x
        case "S":
            if is_min:
                s_min = x
            else:
                s_max = x
        case "V":
            if is_min:
                v_min = x
            else:
                v_max = x
        case _:
            pass


WINDOW_NAME = "HSV Filter"
h_min = 0
s_min = 0
v_min = 0
h_max = 180
s_max = 255
v_max = 255


def main(video_file):
    # Trackbar UI
    cv2.namedWindow(WINDOW_NAME)
    cv2.resizeWindow(WINDOW_NAME, 800, 200)
    cv2.createTrackbar(
        "H Min",
        WINDOW_NAME,
        h_min,
        180,
        partial(update_color_value, color="H", is_min=True),
    )
    cv2.createTrackbar(
        "H Max",
        WINDOW_NAME,
        h_max,
        180,
        partial(update_color_value, color="H", is_min=False),
    )
    cv2.createTrackbar(
        "S Min",
        WINDOW_NAME,
        s_min,
        255,
        partial(update_color_value, color="S", is_min=True),
    )
    cv2.createTrackbar(
        "S Max",
        WINDOW_NAME,
        s_max,
        255,
        partial(update_color_value, color="S", is_min=False),
    )
    cv2.createTrackbar(
        "V Min",
        WINDOW_NAME,
        v_min,
        255,
        partial(update_color_value, color="V", is_min=True),
    )
    cv2.createTrackbar(
        "V Max",
        WINDOW_NAME,
        v_max,
        255,
        partial(update_color_value, color="V", is_min=False),
    )

    # img = cv2.imread(sys.argv[1])  # image file load
    cap = cv2.VideoCapture(video_file)
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Filter in HSV space
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.dilate(mask, None, iterations=1)
            result = cv2.bitwise_and(img, img, mask=mask)

            # Show filtered result
            # combined = np.hstack((img, result))
            # cv2.imshow("HSV Filter Result", cv2.resize(combined, (1280, 600)))
            cv2.imshow("HSV Filter Result", result)

            # Exit on ESC key press
            key = cv2.waitKey(33)
            if key & 0xFF == 27 or key & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Save HSV filter values as a new preset
        print("\n" + "=" * 60)
        print("Saving HSV filter calibration...")
        hsv_preset = {"blue": (h_min, s_min, v_min, h_max, s_max, v_max, 1)}
        save_color_config("color.cfg", hsv_preset)
        print("=" * 60)


if __name__ == "__main__":
    # Cam id 0
    video_file = 0
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    main(video_file)
