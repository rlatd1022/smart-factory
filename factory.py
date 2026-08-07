#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smart Factory Process with PatchCore Vision & HP60C Depth Inspection."""

from __future__ import annotations

import os
import sys
import threading
from argparse import ArgumentParser
from pathlib import Path
from queue import Empty, Queue
from time import sleep, time
from typing import Optional

import cv2
import numpy as np

# smart_factory_edu root
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from iotdemo import FactoryController
from iotdemo.depth import (
    DepthWindowInspector,
    HP60CCamera,
    InspectionResult,
    ShmDepthReader,
)
from iotdemo.vision.wafer_aligner import WaferAligner, WaferAlignedResult
from iotdemo.vision.wafer_tracker import WaferTransitTracker, TrackEvent, TrackerState
from iotdemo.vision.rule_detector import RuleBasedDefectDetector, RuleDefectResult
from iotdemo.deeplearning.anomaly_detector import PatchCoreDetector, AnomalyResult

FORCE_STOP = False


def colorize_depth_frame(
    depth: np.ndarray, min_mm: float = 200.0, max_mm: float = 4000.0
) -> np.ndarray:
    """Colorize a 16-bit depth map into a TURBO colormap BGR image."""
    valid = (depth >= min_mm) & (depth <= max_mm)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip(
        (max_mm - depth[valid]) * 255.0 / (max_mm - min_mm), 0, 255
    ).astype(np.uint8)
    img = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    img[~valid] = (30, 30, 30)
    return img


def thread_cam1(
    q: Queue,
    cam_id: int = 0,
    weights_path: str = "weights/patchcore_resnet18.pkl",
    kick_delay_sec: float = 1.0,
    threshold_override: Optional[float] = None,
    mode: str = "rule",
):
    """Thread 1: Cam1 Vision Inspection (Rule-based HSV/Contour or PatchCore AI) with 1-Shot Tracker & Kicker Delay."""
    aligner = WaferAligner()
    tracker = WaferTransitTracker(center_tolerance_px=65.0, exit_debounce_frames=4, cooldown_sec=0.8)
    rule_detector = RuleBasedDefectDetector()
    patchcore_detector = PatchCoreDetector()

    if mode == "patchcore":
        w_file = _HERE / weights_path
        if w_file.is_file():
            try:
                patchcore_detector.load(str(w_file))
                if threshold_override is not None:
                    patchcore_detector.threshold = threshold_override
                print(f"[Cam1] PatchCore Detector Loaded (Threshold: {patchcore_detector.threshold:.4f})")
            except Exception as e:
                print(f"[Cam1] Error loading PatchCore weights: {e}")
        else:
            print(f"[Cam1] Warning: Weights file not found: {w_file}. Please run `python train_anomaly.py`.")
    else:
        print("[Cam1] Rule-Based HSV/Contour Defect Detector Activated (High Precision & Latency <2ms)")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[Cam1] Warning: Camera {cam_id} cannot be opened.")

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        align_res: WaferAlignedResult = aligner.align_frame(frame)
        track_evt: TrackEvent = tracker.update(align_res, frame.shape)

        disp_frame = align_res.overlay_view.copy()
        cv2.putText(
            disp_frame,
            track_evt.status_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0) if track_evt.state == TrackerState.INSPECTED else (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        q.put(("VIDEO:Cam1 live", disp_frame))

        if track_evt.should_inspect and align_res.aligned_roi is not None:
            try:
                if mode == "rule":
                    rule_res: RuleDefectResult = rule_detector.inspect(align_res.aligned_roi)
                    q.put(("VIDEO:Cam1 Inspection", rule_res.overlay_bgr))

                    if rule_res.is_defect:
                        print(
                            f"[Cam1-RULE] NG: #{track_evt.object_id} {rule_res.description} "
                            f"-> Schedule Kicker 1 in {kick_delay_sec:.1f}s"
                        )
                        threading.Timer(kick_delay_sec, lambda: q.put(("PUSH", 1))).start()
                    else:
                        print(f"[Cam1-RULE] OK: #{track_evt.object_id} PASS (정상 양품)")
                else:
                    anom_res: AnomalyResult = patchcore_detector.predict(align_res.aligned_roi)
                    q.put(("VIDEO:Cam1 Heatmap", anom_res.overlay_bgr))

                    pat = anom_res.pattern
                    if anom_res.is_defect:
                        desc = pat.description if pat else "결함 검출"
                        print(
                            f"[Cam1-AI] NG: #{track_evt.object_id} Score={anom_res.anomaly_score:.2f} ({desc}) "
                            f"-> Schedule Kicker 1 in {kick_delay_sec:.1f}s"
                        )
                        threading.Timer(kick_delay_sec, lambda: q.put(("PUSH", 1))).start()
                    else:
                        print(
                            f"[Cam1-AI] OK: #{track_evt.object_id} Score={anom_res.anomaly_score:.2f} "
                            f"(Threshold: {patchcore_detector.threshold:.2f}) -> PASS"
                        )
            except Exception as e:
                print(f"[Cam1] Inspection error: {e}")

    cap.release()
    q.put(("DONE", None))


def thread_depth_cam(q: Queue, cfg_path: str = "depth_cfg.json"):
    """Thread 2: HP60C Depth Inspection (Minimum Depth & 388mm Evaluation)."""
    inspector = DepthWindowInspector.load_config(cfg_path)
    print(
        f"[Depth] Inspector loaded: Ref={inspector.ref_depth_mm}mm, "
        f"Tol=±{inspector.tol_mm}mm, Kicker={inspector.kicker_num}"
    )

    shm_reader = ShmDepthReader()
    cam_reader: Optional[HP60CCamera] = None
    last_completed_info: Optional[str] = None
    last_completed_status: Optional[str] = None

    while not FORCE_STOP:
        sleep(0.03)
        rgb_frame = None
        depth_frame = None

        if shm_reader.is_available():
            rgb_frame, depth_frame, _ = shm_reader.read()
        elif cam_reader is None:
            try:
                cam_reader = HP60CCamera().open()
            except Exception:
                pass

        if depth_frame is None and cam_reader is not None:
            try:
                depth_frame = cam_reader.read()
            except Exception:
                depth_frame = None

        if depth_frame is None:
            continue

        result: InspectionResult = inspector.process_frame(
            depth_frame, rgb_frame=rgb_frame
        )

        # Trigger kicker on completed NG transit
        if result.event == "OBJECT_DONE":
            if result.should_kick:
                msg = (
                    f"NG: Min={result.min_depth_mm}mm (Ref={result.ref_depth_mm}mm, "
                    f"Δ={result.delta_mm:+.1f}mm) -> KICK Actuator {result.kicker_num}"
                )
                print(f"[Depth] {msg}")
                last_completed_info = msg
                last_completed_status = "NG"
                q.put(("PUSH", result.kicker_num))
            else:
                msg = (
                    f"OK: Min={result.min_depth_mm}mm (Ref={result.ref_depth_mm}mm, "
                    f"Δ={result.delta_mm:+.1f}mm) -> PASS"
                )
                print(f"[Depth] {msg}")
                last_completed_info = msg
                last_completed_status = "OK"

        # Visualization
        view = colorize_depth_frame(depth_frame)
        cx, cy = inspector.resolve_uv(depth_frame.shape)

        # Draw inspection point crosshair & gate circle
        gate_color = (0, 255, 0) if result.mask_ok else (200, 200, 200)
        cv2.drawMarker(
            view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2
        )
        cv2.circle(view, (cx, cy), inspector.mask_radius_px, gate_color, 2)

        # Draw status overlay
        if result.status == "MEASURE":
            status_text = (
                f"MEASURING: {result.current_depth_mm:.0f} mm "
                f"(Min: {result.min_depth_mm:.0f} mm, N={result.samples_count})"
            )
            color = (0, 255, 255)  # Yellow
        elif last_completed_status == "OK":
            status_text = f"LAST: {last_completed_info}"
            color = (0, 255, 0)  # Green
        elif last_completed_status == "NG":
            status_text = f"LAST: {last_completed_info}"
            color = (0, 0, 255)  # Red
        else:
            status_text = "EMPTY (N/A) - Waiting for object at center"
            color = (180, 180, 180)  # Gray

        cv2.putText(
            view,
            f"Target Ref: {inspector.ref_depth_mm:.0f} mm (±{inspector.tol_mm:.0f} mm) | Point: Center ({cx},{cy})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            status_text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        q.put(("VIDEO:Depth live", view))

    if cam_reader is not None:
        cam_reader.close()
    shm_reader.close()
    q.put(("DONE", None))


def imshow(title: str, frame: np.ndarray, pos: Optional[tuple[int, int]] = None):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(
        prog="python3 factory.py",
        description="Smart Factory Orchestrator with Vision & Depth Inspection",
    )
    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port (e.g. /dev/ttyACM0)")
    parser.add_argument("--cam-id", default=0, type=int, help="RGB Camera ID (default 0)")
    parser.add_argument("--depth-cfg", default="depth_cfg.json", type=str, help="Depth config file")
    parser.add_argument("--mode", default="rule", choices=["rule", "patchcore"], type=str, help="Inspection mode: 'rule' (HSV & Contour) or 'patchcore' (DL AI)")
    parser.add_argument("--weights", default="weights/patchcore_resnet18.pkl", type=str, help="PatchCore weights")
    parser.add_argument("--threshold", default=None, type=float, help="PatchCore anomaly score threshold (default: auto from weights)")
    parser.add_argument("--gui", action="store_true", help="Launch integrated Smart Factory GUI Dashboard")
    parser.add_argument("--disable-cam1", action="store_true", help="Disable Cam1 Vision Thread")
    parser.add_argument("--disable-depth", action="store_true", help="Disable Depth Camera Thread")
    parser.add_argument(
        "--kick-delay",
        default=1.0,
        type=float,
        help="Vision kicker 1 trigger delay in seconds (default: 1.0s)",
    )
    parser.add_argument(
        "--pulse-duration",
        default=0.05,
        type=float,
        help="Kicker solenoid ON duration in seconds (default 0.05s / 50ms)",
    )
    parser.add_argument(
        "--no-reverse-actuator",
        action="store_true",
        help="Disable reverse actuator mapping",
    )
    args = parser.parse_args()

    # If --gui is specified, launch the full GUI dashboard directly
    if args.gui:
        import tkinter as tk
        from factory_gui import SmartFactoryDashboard
        root = tk.Tk()
        app = SmartFactoryDashboard(
            root=root,
            device_port=args.device,
            cam_id=args.cam_id,
            depth_cfg=args.depth_cfg,
            weights_path=args.weights,
            threshold_override=args.threshold,
            reverse_actuator=not args.no_reverse_actuator,
            initial_mode=args.mode,
        )
        root.mainloop()
        return

    q: Queue = Queue()
    threads: list[threading.Thread] = []

    # Start Depth Camera thread
    if not args.disable_depth:
        t_depth = threading.Thread(
            target=thread_depth_cam,
            args=(q, args.depth_cfg),
            name="depth_cam",
            daemon=True,
        )
        t_depth.start()
        threads.append(t_depth)

    # Start Cam1 Vision thread
    if not args.disable_cam1:
        t_cam1 = threading.Thread(
            target=thread_cam1,
            args=(q, args.cam_id, args.weights, args.kick_delay, args.threshold, args.mode),
            name="cam1",
            daemon=True,
        )
        t_cam1.start()
        threads.append(t_cam1)

    print("==========================================================")
    print(" Smart Factory Process Started")
    print(f" - Kicker Pulse Duration: {args.pulse_duration * 1000:.0f} ms")
    print(f" - Actuator Mapping: {'Reversed (Vision->Actuator2, Depth->Actuator1)' if not args.no_reverse_actuator else 'Standard'}")
    print(" - Press 'q' in any window to exit")
    print("==========================================================")

    reverse_act = not args.no_reverse_actuator
    with FactoryController(
        conn=FactoryController.Connector.ARDUINO,
        port=args.device,
        debug=True,
        pulse_duration=args.pulse_duration,
        reverse_actuator=reverse_act,
    ) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            try:
                name, data = q.get(timeout=0.1)
            except Empty:
                continue

            if name.startswith("VIDEO:"):
                imshow(name[6:], data)
            elif name == "PUSH":
                kicker_num = int(data)
                print(f"[Actuator] PUSH command received -> Kicking Actuator {kicker_num}")
                ctrl.push_actuator(kicker_num)
            elif name == "DONE":
                FORCE_STOP = True

            q.task_done()

    FORCE_STOP = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Exception in main: {exc}")
        os._exit(0)
