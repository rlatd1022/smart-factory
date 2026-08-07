#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factory_gui.py — 스마트팩토리 실시간 공정 제어 및 다중 비전/결함 경향성 통합 GUI 대시보드."""

from __future__ import annotations

import csv
import datetime
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Add project root to sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from iotdemo.factory_controller import FactoryController
from iotdemo.depth import DepthWindowInspector, HP60CCamera, InspectionResult, ShmDepthReader
from iotdemo.vision.wafer_aligner import WaferAligner, WaferAlignedResult
from iotdemo.vision.wafer_tracker import WaferTransitTracker, TrackEvent, TrackerState, InspectionMode
from iotdemo.vision.rule_detector import RuleBasedDefectDetector, RuleDefectResult, DefectCandidate
from iotdemo.vision.wafer_spatial_analyzer import WaferSpatialAnalyzer, SpatialAnalyticsSummary
from iotdemo.deeplearning.anomaly_detector import PatchCoreDetector, AnomalyResult, PatternAnalysisResult


def colorize_depth(depth: np.ndarray, min_mm: float = 200.0, max_mm: float = 4000.0) -> np.ndarray:
    """16비트 뎁스 맵을 TURBO 컬러맵 BGR 이미지로 변환."""
    valid = (depth >= min_mm) & (depth <= max_mm)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip((max_mm - depth[valid]) * 255.0 / (max_mm - min_mm), 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    img[~valid] = (30, 30, 30)
    return img


class SmartFactoryDashboard:
    """스마트팩토리 일체형 실시간 GUI 대시보드 애플리케이션."""

    def __init__(
        self,
        root: tk.Tk,
        device_port: Optional[str] = None,
        cam_id: int = 0,
        depth_cfg: str = "depth_cfg.json",
        weights_path: str = "weights/patchcore_resnet18.pkl",
        threshold_override: Optional[float] = None,
        reverse_actuator: bool = True,
        initial_mode: str = "rule",
    ):
        self.root = root
        self.root.title("🏭 SMART FACTORY REAL-TIME INSPECTION & CONTROL DASHBOARD")
        self.root.geometry("1540x950")
        self.root.minsize(1024, 600)
        self.root.configure(bg="#12161F")

        self.device_port = device_port
        self.cam_id = cam_id
        self.depth_cfg_path = depth_cfg
        self.weights_path = weights_path
        self.threshold_override = threshold_override
        self.reverse_actuator = reverse_actuator
        self.initial_mode = initial_mode

        # Control Queues & Flags
        self.gui_queue: Queue = Queue()
        self.cmd_queue: Queue = Queue()
        self.running = True

        # Statistics & History
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.history_records: List[dict] = []
        self.score_trend: List[float] = [0.0] * 50
        self.current_threshold = 4.2829
        self.conveyor_pwm = 255
        self.vision_kick_delay_sec = 1.0  # 비전 카메라 -> 키커 1 이동 도달 딜레이 (기본값: 1.0초)
        self.stop_and_inspect_var = tk.BooleanVar(value=True)  # Stop-and-Inspect 모드 활성화 여부
        self.stop_duration_sec = 0.5  # 중심 정지 유지 시간 (초)
        self.camera_brightness = 70  # 카메라 센서 밝기 (기본값: 70)
        self.camera_exposure = 80  # 카메라 노출 시간 (기본값: 80)
        self.insp_mode_var = tk.StringVar(value=initial_mode)  # 'rule' or 'patchcore'

        # Initialize Models, Trackers & Spatial Analyzer
        self.aligner = WaferAligner()
        self.tracker = WaferTransitTracker(
            center_tolerance_px=65.0,
            exit_debounce_frames=4,
            cooldown_sec=0.8,
            mode=InspectionMode.STOP_AND_INSPECT,
        )
        self.rule_detector = RuleBasedDefectDetector()
        self.spatial_analyzer = WaferSpatialAnalyzer(wafer_radius_mm=45.0, grid_size=256)
        self.detector = PatchCoreDetector()
        w_file = _HERE / self.weights_path
        if w_file.is_file():
            try:
                self.detector.load(str(w_file))
                if self.threshold_override is not None:
                    self.detector.threshold = self.threshold_override
                self.current_threshold = self.detector.threshold
            except Exception as e:
                print(f"[GUI] Model load warning: {e}")

        # Hardware Controller
        try:
            self.ctrl = FactoryController(
                conn=FactoryController.Connector.ARDUINO,
                port=self.device_port,
                debug=True,
                reverse_actuator=self.reverse_actuator,
            )
        except Exception as e:
            print(f"[GUI] Controller Init Warning: {e}")
            self.ctrl = None

        # Build GUI Layout
        self._setup_styles()
        self._build_ui()

        # Log Controller status
        if self.ctrl and not self.ctrl.is_dummy:
            self._log(f"[HARDWARE] 아두이노 제어기 정상 연결 완료 ({self.ctrl.port})")
        else:
            self._log("[HARDWARE WARNING] 아두이노 장치가 감지되지 않아 시뮬레이션(가상) 모드로 동작합니다.")

        # Start Background Worker Threads
        self.t_vision = threading.Thread(target=self._worker_vision, daemon=True)
        self.t_depth = threading.Thread(target=self._worker_depth, daemon=True)
        self.t_vision.start()
        self.t_depth.start()

        # Start GUI Refresh Loop (30 FPS)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(33, self._update_gui)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#12161F", foreground="#E2E8F0", font=("Helvetica", 10))
        style.configure("TFrame", background="#12161F")
        style.configure("Card.TFrame", background="#1A202C", relief="flat")
        style.configure("TLabel", background="#12161F", foreground="#E2E8F0")
        style.configure("Card.TLabel", background="#1A202C", foreground="#E2E8F0")
        style.configure("Vertical.TScrollbar", background="#1E293B", troughcolor="#0F172A", borderwidth=0)
        style.configure("Horizontal.TScrollbar", background="#1E293B", troughcolor="#0F172A", borderwidth=0)

    def _build_ui(self):
        # 1. Top Header Bar (Fixed at top)
        header = tk.Frame(self.root, bg="#0F172A", height=60, padx=20, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="🏭 SMART FACTORY REAL-TIME INSPECTION & CONTROL DASHBOARD",
            font=("Helvetica", 15, "bold"),
            bg="#0F172A",
            fg="#38BDF8",
        )
        title_lbl.pack(side=tk.LEFT)

        self.lbl_clock = tk.Label(
            header,
            text="2026-08-06 00:00:00",
            font=("Helvetica", 11),
            bg="#0F172A",
            fg="#94A3B8",
        )
        self.lbl_clock.pack(side=tk.RIGHT, padx=10)

        self.lbl_sys_status = tk.Label(
            header,
            text="● SYSTEM STOPPED",
            font=("Helvetica", 12, "bold"),
            bg="#0F172A",
            fg="#EF4444",
        )
        self.lbl_sys_status.pack(side=tk.RIGHT, padx=20)

        # 2. Scrollable Container Frame (Contains Canvas + Scrollbars)
        scroll_container = tk.Frame(self.root, bg="#12161F")
        scroll_container.pack(fill=tk.BOTH, expand=True)

        self.main_canvas = tk.Canvas(scroll_container, bg="#12161F", highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=self.main_canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(scroll_container, orient="horizontal", command=self.main_canvas.xview)

        self.main_canvas.configure(
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Content frame embedded inside canvas
        content = tk.Frame(self.main_canvas, bg="#12161F", padx=15, pady=15)
        self.canvas_window = self.main_canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_content_configure(event=None):
            bbox = self.main_canvas.bbox("all")
            if bbox:
                x1, y1, x2, y2 = bbox
                cw = max(self.main_canvas.winfo_width(), x2)
                ch = max(y2, content.winfo_reqheight(), 1080)
                self.main_canvas.configure(scrollregion=(0, 0, cw, ch))

        def _on_canvas_configure(event):
            cw = event.width
            rw = max(cw, content.winfo_reqwidth())
            self.main_canvas.itemconfig(self.canvas_window, width=rw)
            _on_content_configure()

        content.bind("<Configure>", _on_content_configure)
        self.main_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if event.num == 4:  # Linux mousewheel up
                self.main_canvas.yview_scroll(-3, "units")
            elif event.num == 5:  # Linux mousewheel down
                self.main_canvas.yview_scroll(3, "units")
            elif hasattr(event, "delta") and event.delta:  # Windows/macOS
                direction = -1 if event.delta > 0 else 1
                self.main_canvas.yview_scroll(direction * 3, "units")

        def _on_shift_mousewheel(event):
            if event.num == 4:
                self.main_canvas.xview_scroll(-3, "units")
            elif event.num == 5:
                self.main_canvas.xview_scroll(3, "units")
            elif hasattr(event, "delta") and event.delta:
                direction = -1 if event.delta > 0 else 1
                self.main_canvas.xview_scroll(direction * 3, "units")

        self.root.bind_all("<Button-4>", _on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", _on_mousewheel, add="+")
        self.root.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        self.root.bind_all("<Shift-Button-4>", _on_shift_mousewheel, add="+")
        self.root.bind_all("<Shift-Button-5>", _on_shift_mousewheel, add="+")
        self.root.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel, add="+")

        # 3. Main Content Split (Left: Multi-Vision 60%, Right: Controls & Analytics 40%)
        left_panel = tk.Frame(content, bg="#12161F")
        left_panel.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 10))

        right_panel = tk.Frame(content, bg="#12161F", width=480)
        right_panel.pack(fill=tk.Y, side=tk.RIGHT, padx=(10, 0))

        # --- LEFT PANEL: 3 Multi-Vision Feeds ---
        v_card = tk.LabelFrame(
            left_panel,
            text=" 📷 Real-time Vision Feeds (Live RGB / PatchCore Heatmap / Depth Map) ",
            font=("Helvetica", 11, "bold"),
            bg="#1A202C",
            fg="#38BDF8",
            padx=10,
            pady=10,
        )
        v_card.pack(fill=tk.BOTH, expand=True)

        # Top 2 feeds (RGB & Heatmap)
        feeds_top = tk.Frame(v_card, bg="#1A202C")
        feeds_top.pack(fill=tk.BOTH, expand=True)

        f1 = tk.Frame(feeds_top, bg="#111827", padx=5, pady=5)
        f1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(f1, text="1. RGB Cam & 12 O'Clock Alignment", font=("Helvetica", 10, "bold"), bg="#111827", fg="#38BDF8").pack(anchor=tk.W)
        self.canvas_rgb = tk.Label(f1, bg="#000000")
        self.canvas_rgb.pack(fill=tk.BOTH, expand=True, pady=4)

        f2 = tk.Frame(feeds_top, bg="#111827", padx=5, pady=5)
        f2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(f2, text="2. PatchCore AI Defect Heatmap", font=("Helvetica", 10, "bold"), bg="#111827", fg="#F59E0B").pack(anchor=tk.W)
        self.canvas_heat = tk.Label(f2, bg="#000000")
        self.canvas_heat.pack(fill=tk.BOTH, expand=True, pady=4)

        # Bottom feed (Depth) & Log
        feeds_bot = tk.Frame(v_card, bg="#1A202C")
        feeds_bot.pack(fill=tk.BOTH, expand=True)

        f3 = tk.Frame(feeds_bot, bg="#111827", padx=5, pady=5)
        f3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(f3, text="3. HP60C Depth TURBO Colormap", font=("Helvetica", 10, "bold"), bg="#111827", fg="#10B981").pack(anchor=tk.W)
        self.canvas_depth = tk.Label(f3, bg="#000000")
        self.canvas_depth.pack(fill=tk.BOTH, expand=True, pady=4)

        f4 = tk.Frame(feeds_bot, bg="#111827", padx=5, pady=5)
        f4.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(f4, text="4. Real-time Inspection Log & Polar Coords", font=("Helvetica", 10, "bold"), bg="#111827", fg="#A855F7").pack(anchor=tk.W)
        
        self.txt_log = tk.Text(f4, bg="#0A0F1D", fg="#CBD5E1", font=("Courier", 9), height=10, relief="flat")
        self.txt_log.pack(fill=tk.BOTH, expand=True, pady=4)

        # --- RIGHT PANEL: Controls, Statistics, Pattern Analytics, Trend ---
        
        # 1. Process Control Panel
        ctrl_card = tk.LabelFrame(
            right_panel,
            text=" ⚙️ Process & Conveyor Control ",
            font=("Helvetica", 11, "bold"),
            bg="#1A202C",
            fg="#38BDF8",
            padx=12,
            pady=8,
        )
        ctrl_card.pack(fill=tk.X, pady=(0, 10))

        btn_box = tk.Frame(ctrl_card, bg="#1A202C")
        btn_box.pack(fill=tk.X, pady=4)

        self.btn_start = tk.Button(
            btn_box,
            text="▶ START",
            font=("Helvetica", 10, "bold"),
            bg="#10B981",
            fg="#FFFFFF",
            activebackground="#059669",
            command=self._cmd_start,
            width=9,
            relief="flat",
        )
        self.btn_start.pack(side=tk.LEFT, padx=2)

        self.btn_stop = tk.Button(
            btn_box,
            text="⏹ STOP",
            font=("Helvetica", 10, "bold"),
            bg="#EF4444",
            fg="#FFFFFF",
            activebackground="#DC2626",
            command=self._cmd_stop,
            width=9,
            relief="flat",
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.btn_kick1 = tk.Button(
            btn_box,
            text="⚡ KICK 1",
            font=("Helvetica", 9, "bold"),
            bg="#6366F1",
            fg="#FFFFFF",
            command=lambda: self._cmd_kick(1),
            width=8,
            relief="flat",
        )
        self.btn_kick1.pack(side=tk.LEFT, padx=2)

        self.btn_kick2 = tk.Button(
            btn_box,
            text="⚡ KICK 2",
            font=("Helvetica", 9, "bold"),
            bg="#8B5CF6",
            fg="#FFFFFF",
            command=lambda: self._cmd_kick(2),
            width=8,
            relief="flat",
        )
        self.btn_kick2.pack(side=tk.LEFT, padx=2)

        # Inspection Mode Toggle (Stop & Inspect vs Continuous Rollback)
        mode_box = tk.Frame(ctrl_card, bg="#1A202C")
        mode_box.pack(fill=tk.X, pady=(6, 2))

        self.chk_stop_mode = tk.Checkbutton(
            mode_box,
            text="⏹️ Stop & Inspect Mode (중심 정지 검사)",
            variable=self.stop_and_inspect_var,
            command=self._cmd_toggle_mode,
            font=("Helvetica", 9, "bold"),
            bg="#1A202C",
            fg="#A7F3D0",
            selectcolor="#064E3B",
            activebackground="#1A202C",
            activeforeground="#34D399",
        )
        self.chk_stop_mode.pack(anchor=tk.W)

        # Stop Duration Slider
        dur_box = tk.Frame(ctrl_card, bg="#1A202C")
        dur_box.pack(fill=tk.X, pady=(2, 2))

        self.lbl_stop_dur = tk.Label(
            dur_box,
            text=f"Stop Duration: {self.stop_duration_sec:.1f} s (정지 안정화 시간)",
            font=("Helvetica", 8),
            bg="#1A202C",
            fg="#94A3B8",
        )
        self.lbl_stop_dur.pack(anchor=tk.W)

        self.slider_stop_dur = tk.Scale(
            dur_box,
            from_=0.2,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg="#1E293B",
            fg="#A7F3D0",
            troughcolor="#0F172A",
            highlightthickness=0,
            command=self._cmd_stop_dur_change,
        )
        self.slider_stop_dur.set(self.stop_duration_sec)
        self.slider_stop_dur.pack(fill=tk.X, pady=1)

        # Speed Slider
        speed_box = tk.Frame(ctrl_card, bg="#1A202C")
        speed_box.pack(fill=tk.X, pady=(4, 2))

        self.lbl_speed = tk.Label(speed_box, text="Conveyor Speed: 100% (255 PWM)", font=("Helvetica", 9, "bold"), bg="#1A202C", fg="#38BDF8")
        self.lbl_speed.pack(anchor=tk.W)

        self.slider_speed = tk.Scale(
            speed_box,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            bg="#1E293B",
            fg="#38BDF8",
            troughcolor="#0F172A",
            highlightthickness=0,
            command=self._cmd_speed_change,
        )
        self.slider_speed.set(255)
        self.slider_speed.pack(fill=tk.X, pady=2)

        # Kicker Delay Slider
        delay_box = tk.Frame(ctrl_card, bg="#1A202C")
        delay_box.pack(fill=tk.X, pady=(4, 2))

        self.lbl_delay = tk.Label(delay_box, text="Vision Kicker 1 Delay: 1.00 s", font=("Helvetica", 9, "bold"), bg="#1A202C", fg="#F59E0B")
        self.lbl_delay.pack(anchor=tk.W)

        self.slider_delay = tk.Scale(
            delay_box,
            from_=0.1,
            to=5.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            bg="#1E293B",
            fg="#F59E0B",
            troughcolor="#0F172A",
            highlightthickness=0,
            command=self._cmd_delay_change,
        )
        self.slider_delay.set(1.0)
        self.slider_delay.pack(fill=tk.X, pady=2)

        # Inspection Engine Selector (Rule-based vs PatchCore AI)
        engine_box = tk.LabelFrame(
            ctrl_card,
            text=" 🔬 Vision Engine (비전 검사 엔진 선택) ",
            font=("Helvetica", 9, "bold"),
            bg="#1A202C",
            fg="#A78BFA",
            padx=6,
            pady=4,
        )
        engine_box.pack(fill=tk.X, pady=(4, 2))

        rb_rule = tk.Radiobutton(
            engine_box,
            text="⚡ 룰베이스 (Rule-based HSV & 윤곽선)",
            variable=self.insp_mode_var,
            value="rule",
            bg="#1A202C",
            fg="#34D399",
            selectcolor="#0F172A",
            activebackground="#1A202C",
            activeforeground="#34D399",
            font=("Helvetica", 8, "bold"),
            command=self._cmd_engine_change,
        )
        rb_rule.pack(anchor=tk.W)

        rb_ai = tk.Radiobutton(
            engine_box,
            text="🧠 딥러닝 AI (PatchCore Anomaly)",
            variable=self.insp_mode_var,
            value="patchcore",
            bg="#1A202C",
            fg="#F472B6",
            selectcolor="#0F172A",
            activebackground="#1A202C",
            activeforeground="#F472B6",
            font=("Helvetica", 8, "bold"),
            command=self._cmd_engine_change,
        )
        rb_ai.pack(anchor=tk.W)

        # Defect Threshold / Area Slider
        thresh_box = tk.Frame(ctrl_card, bg="#1A202C")
        thresh_box.pack(fill=tk.X, pady=(4, 2))

        self.lbl_thresh = tk.Label(
            thresh_box,
            text="Min Defect Area: 12 px (최소 불량 면적)" if self.insp_mode_var.get() == "rule" else f"AI Defect Threshold: {self.current_threshold:.2f}",
            font=("Helvetica", 9, "bold"),
            bg="#1A202C",
            fg="#EC4899",
        )
        self.lbl_thresh.pack(anchor=tk.W)

        self.slider_thresh = tk.Scale(
            thresh_box,
            from_=2.0 if self.insp_mode_var.get() == "rule" else 1.0,
            to=100.0 if self.insp_mode_var.get() == "rule" else 20.0,
            resolution=1.0 if self.insp_mode_var.get() == "rule" else 0.1,
            orient=tk.HORIZONTAL,
            bg="#1E293B",
            fg="#EC4899",
            troughcolor="#0F172A",
            highlightthickness=0,
            command=self._cmd_threshold_change,
        )
        self.slider_thresh.set(12.0 if self.insp_mode_var.get() == "rule" else self.current_threshold)
        self.slider_thresh.pack(fill=tk.X, pady=2)

        # 📷 Camera Sensor & Lighting Section (카메라 비전 조명/밝기/노출 조절)
        cam_light_box = tk.LabelFrame(
            ctrl_card,
            text=" 📷 Camera Lighting & Sensor Controls (비전 조명/센서 조절) ",
            font=("Helvetica", 9, "bold"),
            bg="#1A202C",
            fg="#38BDF8",
            padx=8,
            pady=4,
        )
        cam_light_box.pack(fill=tk.X, pady=(6, 2))

        # Camera Brightness Slider
        self.lbl_bright = tk.Label(
            cam_light_box,
            text=f"Camera Brightness: {self.camera_brightness} (센서 밝기)",
            font=("Helvetica", 8),
            bg="#1A202C",
            fg="#94A3B8",
        )
        self.lbl_bright.pack(anchor=tk.W, pady=(2, 0))

        self.slider_bright = tk.Scale(
            cam_light_box,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            bg="#1E293B",
            fg="#38BDF8",
            troughcolor="#0F172A",
            highlightthickness=0,
            command=self._cmd_brightness_change,
        )
        self.slider_bright.set(self.camera_brightness)
        self.slider_bright.pack(fill=tk.X, pady=1)

        # Camera Exposure Slider
        self.lbl_exposure = tk.Label(
            cam_light_box,
            text=f"Camera Exposure: {self.camera_exposure} (노출 시간)",
            font=("Helvetica", 8),
            bg="#1A202C",
            fg="#94A3B8",
        )
        self.lbl_exposure.pack(anchor=tk.W, pady=(2, 0))

        self.slider_exposure = tk.Scale(
            cam_light_box,
            from_=3,
            to=500,
            orient=tk.HORIZONTAL,
            bg="#1E293B",
            fg="#F59E0B",
            troughcolor="#0F172A",
            highlightthickness=0,
            command=self._cmd_exposure_change,
        )
        self.slider_exposure.set(self.camera_exposure)
        self.slider_exposure.pack(fill=tk.X, pady=1)

        # Quick Preset Buttons
        preset_row = tk.Frame(cam_light_box, bg="#1A202C")
        preset_row.pack(fill=tk.X, pady=(3, 2))

        tk.Button(
            preset_row,
            text="🔅 반사 방지 (저조도)",
            font=("Helvetica", 8, "bold"),
            bg="#1E293B",
            fg="#38BDF8",
            padx=4,
            pady=2,
            relief="flat",
            command=lambda: self._cmd_set_cam_preset("DIM"),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        tk.Button(
            preset_row,
            text="🌙 최소 밝기 (어둡게)",
            font=("Helvetica", 8),
            bg="#1E293B",
            fg="#FCD34D",
            padx=4,
            pady=2,
            relief="flat",
            command=lambda: self._cmd_set_cam_preset("DARK"),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        tk.Button(
            preset_row,
            text="🔆 기본 표준 (128)",
            font=("Helvetica", 8),
            bg="#1E293B",
            fg="#CBD5E1",
            padx=4,
            pady=2,
            relief="flat",
            command=lambda: self._cmd_set_cam_preset("NORMAL"),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # 2. Quality Statistics Card
        stat_card = tk.LabelFrame(
            right_panel,
            text=" 📊 Real-time Quality & Yield Statistics ",
            font=("Helvetica", 11, "bold"),
            bg="#1A202C",
            fg="#38BDF8",
            padx=12,
            pady=8,
        )
        stat_card.pack(fill=tk.X, pady=(0, 10))

        grid_box = tk.Frame(stat_card, bg="#1A202C")
        grid_box.pack(fill=tk.X, pady=2)

        self.lbl_total = self._make_stat_tile(grid_box, "TOTAL", "0 EA", "#38BDF8", 0, 0)
        self.lbl_ok = self._make_stat_tile(grid_box, "PASS (OK)", "0 EA", "#10B981", 0, 1)
        self.lbl_ng = self._make_stat_tile(grid_box, "DEFECT (NG)", "0 EA", "#EF4444", 1, 0)
        self.lbl_yield = self._make_stat_tile(grid_box, "YIELD RATE", "100.0 %", "#F59E0B", 1, 1)

        # 3. Virtual Circular Wafer Spatial Heatmap & Defect Probability Card
        pat_card = tk.LabelFrame(
            right_panel,
            text=" 🪐 Virtual Wafer Spatial Defect Heatmap & Probability (가상 원형 판 불량 확률) ",
            font=("Helvetica", 10, "bold"),
            bg="#1A202C",
            fg="#38BDF8",
            padx=10,
            pady=6,
        )
        pat_card.pack(fill=tk.X, pady=(0, 8))

        # Main Split: [Left: 220x220 Virtual Wafer Map] | [Right: Sector Probabilities & Metrics]
        map_split = tk.Frame(pat_card, bg="#1A202C")
        map_split.pack(fill=tk.X, pady=2)

        # Left: Virtual Circular Wafer Canvas
        map_frame = tk.Frame(map_split, bg="#0F172A", padx=2, pady=2, relief="groove")
        map_frame.pack(side=tk.LEFT, padx=(0, 6))
        self.canvas_wafer_map = tk.Label(map_frame, bg="#0F172A")
        self.canvas_wafer_map.pack(fill=tk.BOTH, expand=True)

        # Right: Real-time Sector Probability Metrics
        prob_box = tk.Frame(map_split, bg="#1A202C")
        prob_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(prob_box, text="📍 구역별 불량 발생 확률 (P)", font=("Helvetica", 8, "bold"), bg="#1A202C", fg="#A78BFA").pack(anchor=tk.W)

        # Quadrant Probabilities (Q1~Q4)
        quad_grid = tk.Frame(prob_box, bg="#1A202C")
        quad_grid.pack(fill=tk.X, pady=2)

        self.lbl_prob_q1 = tk.Label(quad_grid, text="Q1(NE): 0.0%", font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#38BDF8", padx=4, pady=2)
        self.lbl_prob_q1.grid(row=0, column=0, sticky="ew", padx=1, pady=1)

        self.lbl_prob_q2 = tk.Label(quad_grid, text="Q2(SE): 0.0%", font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#38BDF8", padx=4, pady=2)
        self.lbl_prob_q2.grid(row=0, column=1, sticky="ew", padx=1, pady=1)

        self.lbl_prob_q3 = tk.Label(quad_grid, text="Q3(SW): 0.0%", font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#38BDF8", padx=4, pady=2)
        self.lbl_prob_q3.grid(row=1, column=0, sticky="ew", padx=1, pady=1)

        self.lbl_prob_q4 = tk.Label(quad_grid, text="Q4(NW): 0.0%", font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#38BDF8", padx=4, pady=2)
        self.lbl_prob_q4.grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        quad_grid.grid_columnconfigure(0, weight=1)
        quad_grid.grid_columnconfigure(1, weight=1)

        # Radial Zone Probabilities
        self.lbl_rad_dist = tk.Label(
            prob_box,
            text="Center: 0% | Mid: 0% | Edge: 0%",
            font=("Helvetica", 7, "bold"),
            bg="#111827",
            fg="#FCD34D",
            padx=4,
            pady=2,
        )
        self.lbl_rad_dist.pack(fill=tk.X, pady=(2, 4))

        # Reset Map Button
        btn_reset_map = tk.Button(
            prob_box,
            text="🔄 맵 초기화",
            font=("Helvetica", 7, "bold"),
            bg="#334155",
            fg="#E2E8F0",
            padx=4,
            pady=1,
            relief="flat",
            command=self._cmd_reset_spatial_map,
        )
        btn_reset_map.pack(fill=tk.X)

        # Bottom info: Severity & Real-time Tendency AI Diagnosis
        diag_box = tk.Frame(pat_card, bg="#0F172A", padx=6, pady=4)
        diag_box.pack(fill=tk.X, pady=(4, 0))

        row1 = tk.Frame(diag_box, bg="#0F172A")
        row1.pack(fill=tk.X, pady=1)
        tk.Label(row1, text="최근 검사 등급:", font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#94A3B8").pack(side=tk.LEFT)
        self.lbl_severity = tk.Label(row1, text="NORMAL (양품)", font=("Helvetica", 9, "bold"), bg="#10B981", fg="#FFFFFF", padx=6, pady=1)
        self.lbl_severity.pack(side=tk.RIGHT)

        row2 = tk.Frame(diag_box, bg="#0F172A")
        row2.pack(fill=tk.X, pady=1)
        tk.Label(row2, text="결함 패턴 유형:", font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#94A3B8").pack(side=tk.LEFT)
        self.lbl_pattern = tk.Label(row2, text="NONE", font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#38BDF8")
        self.lbl_pattern.pack(side=tk.RIGHT)

        self.lbl_tendency_diag = tk.Label(
            diag_box,
            text="🟢 [정상] 누적 결함 없음 (청정 상태 유지)",
            font=("Helvetica", 8, "bold"),
            bg="#0F172A",
            fg="#10B981",
            wraplength=440,
            justify=tk.LEFT,
        )
        self.lbl_tendency_diag.pack(fill=tk.X, pady=(2, 0))

        # 4. Live Trend Chart & Export Buttons
        trend_card = tk.LabelFrame(
            right_panel,
            text=" 📈 Real-time Anomaly Score Trend ",
            font=("Helvetica", 11, "bold"),
            bg="#1A202C",
            fg="#38BDF8",
            padx=12,
            pady=8,
        )
        trend_card.pack(fill=tk.BOTH, expand=True)

        self.chart_canvas = tk.Canvas(trend_card, bg="#0F172A", height=120, highlightthickness=0)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True, pady=4)

        act_box = tk.Frame(trend_card, bg="#1A202C")
        act_box.pack(fill=tk.X, pady=4)

        btn_csv = tk.Button(
            act_box,
            text="📥 Export CSV Report",
            font=("Helvetica", 9, "bold"),
            bg="#0284C7",
            fg="#FFFFFF",
            command=self._cmd_export_csv,
            relief="flat",
        )
        btn_csv.pack(side=tk.LEFT, padx=3)

        btn_reset = tk.Button(
            act_box,
            text="🔄 Reset Counts",
            font=("Helvetica", 9),
            bg="#475569",
            fg="#FFFFFF",
            command=self._cmd_reset_counts,
            relief="flat",
        )
        btn_reset.pack(side=tk.RIGHT, padx=3)

    def _make_stat_tile(self, parent, title: str, init_val: str, color: str, r: int, c: int) -> tk.Label:
        box = tk.Frame(parent, bg="#0F172A", padx=8, pady=6)
        box.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
        parent.grid_columnconfigure(c, weight=1)
        tk.Label(box, text=title, font=("Helvetica", 8, "bold"), bg="#0F172A", fg="#94A3B8").pack(anchor=tk.W)
        lbl = tk.Label(box, text=init_val, font=("Helvetica", 12, "bold"), bg="#0F172A", fg=color)
        lbl.pack(anchor=tk.E)
        return lbl

    # --- Hardware Control Callbacks ---

    def _cmd_start(self):
        if self.ctrl:
            self.ctrl.system_start()
            self.ctrl.conveyor_start(self.conveyor_pwm)
        self.lbl_sys_status.config(text="● SYSTEM RUNNING", fg="#10B981")
        self._log(f"[SYSTEM] 공정 가동 시작 (Speed={self.conveyor_pwm} PWM)")

    def _cmd_stop(self):
        if self.ctrl:
            self.ctrl.system_stop()
        self.lbl_sys_status.config(text="● SYSTEM STOPPED", fg="#EF4444")
        self._log("[SYSTEM] 공정 정지 (Stop)")

    def _cmd_toggle_mode(self):
        enabled = self.stop_and_inspect_var.get()
        mode = InspectionMode.STOP_AND_INSPECT if enabled else InspectionMode.CONTINUOUS
        self.tracker.set_mode(mode)
        mode_str = "STOP & INSPECT (중심 정지 정밀 검사)" if enabled else "CONTINUOUS (연속 이송 검사)"
        self._log(f"[MODE] 검사 방식 전환: {mode_str}")

    def _cmd_stop_dur_change(self, val):
        self.stop_duration_sec = float(val)
        self.lbl_stop_dur.config(text=f"Stop Duration: {self.stop_duration_sec:.1f} s (정지 안정화 시간)")

    def _cmd_speed_change(self, val):
        self.conveyor_pwm = int(val)
        pct = (self.conveyor_pwm * 100) // 255
        self.lbl_speed.config(text=f"Conveyor Speed: {pct}% ({self.conveyor_pwm} PWM)")
        if self.ctrl:
            self.ctrl.conveyor_speed = self.conveyor_pwm

    def _cmd_delay_change(self, val):
        self.vision_kick_delay_sec = float(val)
        self.lbl_delay.config(text=f"Vision Kicker 1 Delay: {self.vision_kick_delay_sec:.2f} s")

    def _cmd_engine_change(self):
        engine = self.insp_mode_var.get()
        if engine == "rule":
            self.slider_thresh.config(from_=2.0, to=100.0, resolution=1.0)
            self.slider_thresh.set(self.rule_detector.min_defect_area)
            if hasattr(self, "lbl_thresh"):
                self.lbl_thresh.config(text=f"Min Defect Area: {self.rule_detector.min_defect_area:.0f} px (최소 불량 면적)")
            self._log("[ENGINE] 비전 검사 엔진 전환 -> ⚡ 룰베이스 (Rule-based HSV & 윤곽선)")
        else:
            self.slider_thresh.config(from_=1.0, to=20.0, resolution=0.1)
            self.slider_thresh.set(self.current_threshold)
            if hasattr(self, "lbl_thresh"):
                self.lbl_thresh.config(text=f"AI Defect Threshold: {self.current_threshold:.2f} (불량 판정 임계값)")
            self._log("[ENGINE] 비전 검사 엔진 전환 -> 🧠 딥러닝 AI (PatchCore Anomaly)")

    def _cmd_threshold_change(self, val):
        val_f = float(val)
        if self.insp_mode_var.get() == "rule":
            self.rule_detector.min_defect_area = val_f
            if hasattr(self, "lbl_thresh"):
                self.lbl_thresh.config(text=f"Min Defect Area: {val_f:.0f} px (최소 불량 면적)")
            self._log(f"[RULE THRESHOLD] 최소 결함 면적 변경: {val_f:.0f} px")
        else:
            self.current_threshold = val_f
            self.detector.threshold = self.current_threshold
            if hasattr(self, "lbl_thresh"):
                self.lbl_thresh.config(text=f"AI Defect Threshold: {self.current_threshold:.2f} (불량 판정 임계값)")
            self._log(f"[AI THRESHOLD] 불량 판정 임계값 변경: {self.current_threshold:.2f}")

    def _cmd_brightness_change(self, val):
        self.camera_brightness = int(val)
        if hasattr(self, "lbl_bright"):
            self.lbl_bright.config(text=f"Camera Brightness: {self.camera_brightness} (센서 밝기)")
        try:
            subprocess.run(
                ["v4l2-ctl", f"-d=/dev/video{self.cam_id}", f"--set-ctrl=brightness={self.camera_brightness}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def _cmd_exposure_change(self, val):
        self.camera_exposure = int(val)
        if hasattr(self, "lbl_exposure"):
            self.lbl_exposure.config(text=f"Camera Exposure: {self.camera_exposure} (노출 시간)")
        try:
            subprocess.run(
                [
                    "v4l2-ctl",
                    f"-d=/dev/video{self.cam_id}",
                    "--set-ctrl=exposure_auto=1",
                    f"--set-ctrl=exposure_absolute={self.camera_exposure}",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def _cmd_set_cam_preset(self, mode: str):
        if mode == "DARK":
            if hasattr(self, "slider_bright"):
                self.slider_bright.set(30)
            if hasattr(self, "slider_exposure"):
                self.slider_exposure.set(40)
            self._cmd_brightness_change(30)
            self._cmd_exposure_change(40)
            self._log("[CAMERA] 조명/센서 프리셋: 최소 밝기 모드 (밝기: 30, 노출: 40)")
        elif mode == "DIM":
            if hasattr(self, "slider_bright"):
                self.slider_bright.set(70)
            if hasattr(self, "slider_exposure"):
                self.slider_exposure.set(80)
            self._cmd_brightness_change(70)
            self._cmd_exposure_change(80)
            self._log("[CAMERA] 조명/센서 프리셋: 반사 방지 저조도 모드 (밝기: 70, 노출: 80)")
        elif mode == "NORMAL":
            if hasattr(self, "slider_bright"):
                self.slider_bright.set(128)
            if hasattr(self, "slider_exposure"):
                self.slider_exposure.set(128)
            self._cmd_brightness_change(128)
            self._cmd_exposure_change(128)
            self._log("[CAMERA] 조명/센서 프리셋: 기본 표준 모드 (밝기: 128, 노출: 128)")

    def _cmd_kick(self, kicker_num: int):
        if self.ctrl:
            self.ctrl.push_actuator(kicker_num)
        self._log(f"[ACTUATOR] 수동 키커 {kicker_num} 즉시 작동")

    def _cmd_reset_spatial_map(self):
        self.spatial_analyzer.reset()
        self._update_spatial_ui()
        self._log("[SPATIAL MAP] 가상 원형 웨이퍼 결함 밀도 및 확률 맵 초기화 완료")

    def _cmd_reset_counts(self):
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.score_trend = [0.0] * 50
        self.history_records.clear()
        self.tracker.reset()
        self.spatial_analyzer.reset()
        self._update_stat_ui()
        self._update_spatial_ui()
        self._log("[RESET] 검사 수량, 이력 및 공간 확률 맵 초기화 완료")

    def _cmd_export_csv(self):
        if not self.history_records:
            messagebox.showinfo("Export CSV", "저장할 검사 이력이 없습니다.")
            return

        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"inspection_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if not fpath:
            return

        try:
            with open(fpath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Verdict", "AnomalyScore", "Threshold", "Severity", "PatternType", "DefectCount", "Dispersion_mm", "PolarCoords"])
                for r in self.history_records:
                    writer.writerow([
                        r.get("timestamp"),
                        r.get("verdict"),
                        r.get("score"),
                        r.get("threshold"),
                        r.get("severity"),
                        r.get("pattern"),
                        r.get("defects"),
                        r.get("dispersion"),
                        r.get("polar"),
                    ])
            messagebox.showinfo("Export CSV", f"성공적으로 저장되었습니다:\n{fpath}")
            self._log(f"[EXPORT] CSV 리포트 저장 완료: {fpath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"CSV 저장 중 오류 발생: {e}")

    def _log(self, text: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{now}] {text}\n"
        self.txt_log.insert(tk.END, line)
        self.txt_log.see(tk.END)

    def _update_stat_ui(self):
        self.lbl_total.config(text=f"{self.total_count} EA")
        self.lbl_ok.config(text=f"{self.ok_count} EA")
        self.lbl_ng.config(text=f"{self.ng_count} EA")
        rate = (self.ok_count / max(self.total_count, 1)) * 100.0
        self.lbl_yield.config(text=f"{rate:.1f} %")

    def _update_spatial_ui(self):
        summary = self.spatial_analyzer.get_summary()
        qp = summary.quadrant_prob

        def _get_q_color(prob_val: float) -> str:
            if prob_val >= 25.0:
                return "#EF4444"
            elif prob_val >= 10.0:
                return "#F59E0B"
            else:
                return "#38BDF8"

        if hasattr(self, "lbl_prob_q1"):
            self.lbl_prob_q1.config(text=f"Q1(NE): {qp.get('Q1_NE', 0.0):.1f}%", fg=_get_q_color(qp.get('Q1_NE', 0.0)))
            self.lbl_prob_q2.config(text=f"Q2(SE): {qp.get('Q2_SE', 0.0):.1f}%", fg=_get_q_color(qp.get('Q2_SE', 0.0)))
            self.lbl_prob_q3.config(text=f"Q3(SW): {qp.get('Q3_SW', 0.0):.1f}%", fg=_get_q_color(qp.get('Q3_SW', 0.0)))
            self.lbl_prob_q4.config(text=f"Q4(NW): {qp.get('Q4_NW', 0.0):.1f}%", fg=_get_q_color(qp.get('Q4_NW', 0.0)))

        rp = summary.radial_prob
        if hasattr(self, "lbl_rad_dist"):
            self.lbl_rad_dist.config(
                text=f"Center: {rp.get('CENTER', 0.0):.1f}% | Mid: {rp.get('MID', 0.0):.1f}% | Edge: {rp.get('EDGE', 0.0):.1f}%"
            )

        if hasattr(self, "lbl_tendency_diag"):
            diag_color = "#EF4444" if summary.is_hotspot_detected else ("#10B981" if summary.total_defects == 0 else "#F59E0B")
            self.lbl_tendency_diag.config(text=summary.tendency_diagnosis, fg=diag_color)

    # --- Background Worker Threads ---

    def _worker_vision(self):
        """웹캠 비전 검사 스레드 (Stop-and-Inspect 정밀 정지 검사 + On-the-Fly 연속 검사 롤백 지원)."""
        cap = cv2.VideoCapture(self.cam_id)

        while self.running:
            try:
                time.sleep(0.03)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # 1. Wafer Detection & 12-o'clock Alignment
                align_res: WaferAlignedResult = self.aligner.align_frame(frame)

                # 2. Update Object Transit State Machine
                track_evt: TrackEvent = self.tracker.update(align_res, frame.shape)

                heat_view = None

                # 3-A. [Stop-and-Inspect Mode Sequence]
                if track_evt.should_stop_conveyor and self.stop_and_inspect_var.get():
                    self.gui_queue.put(("LOG", f"[STOP & INSPECT] 웨이퍼 #{track_evt.object_id} 화각 중심 도달 -> 컨베이어 정지"))

                    # 1) 컨베이어 즉시 일시 정지
                    if self.ctrl and self.running:
                        self.ctrl.conveyor_stop()

                    # 2) 기계 진동 및 잔상 감쇠 대기 (150ms)
                    time.sleep(0.15)

                    # 3) 카메라 버퍼 플러시 (이동 중 잔상 제거) 및 정지 고화질 프레임 획득
                    for _ in range(3):
                        cap.grab()
                    clean_ret, clean_frame = cap.read()
                    if clean_ret and clean_frame is not None:
                        align_res = self.aligner.align_frame(clean_frame)

                    # 4) 정지 프레임 기반 1회 정밀 비전 결함 검사
                    if align_res.aligned_roi is not None:
                        try:
                            if self.insp_mode_var.get() == "rule":
                                rule_res: RuleDefectResult = self.rule_detector.inspect(align_res.aligned_roi)
                                heat_view = rule_res.overlay_bgr
                                self.gui_queue.put(("INSPECT_RULE_EVENT", (track_evt.object_id, rule_res)))
                                is_defect = rule_res.is_defect
                            else:
                                anom_res: AnomalyResult = self.detector.predict(align_res.aligned_roi)
                                heat_view = anom_res.overlay_bgr
                                self.gui_queue.put(("INSPECT_EVENT", (track_evt.object_id, anom_res)))
                                is_defect = anom_res.is_defect

                            # 5) 불량품 배출 지연 예약
                            if is_defect:
                                delay = self.vision_kick_delay_sec
                                obj_id = track_evt.object_id

                                def _delayed_kick(kicker_num=1, target_id=obj_id):
                                    if self.ctrl and self.running:
                                        self.ctrl.push_actuator(kicker_num)
                                    self.gui_queue.put(("LOG", f"[KICKER 1] 웨이퍼 #{target_id} 배출 완료"))

                                threading.Timer(delay, _delayed_kick).start()
                                self.gui_queue.put(("LOG", f"[KICKER 1] 웨이퍼 #{obj_id} 불량 배출 예약 ({delay:.2f}초 후 도달)"))
                        except Exception as e:
                            self.gui_queue.put(("LOG", f"[VISION ERROR] 검사 오류: {e}"))

                    # 6) 설정된 잔여 정지 시간 대기
                    rem_time = max(self.stop_duration_sec - 0.15, 0.1)
                    time.sleep(rem_time)

                    # 7) 컨베이어 재가동 및 상태 전이
                    if self.ctrl and self.running:
                        self.ctrl.conveyor_start(self.conveyor_pwm)
                    self.tracker.notify_resumed()
                    self.gui_queue.put(("LOG", f"[STOP & INSPECT] 웨이퍼 #{track_evt.object_id} 검사 완료 -> 라인 재가동"))

                # 3-B. [Continuous On-the-Fly Inspection Mode (Rollback/Continuous)]
                elif track_evt.should_inspect and align_res.aligned_roi is not None:
                    try:
                        if self.insp_mode_var.get() == "rule":
                            rule_res: RuleDefectResult = self.rule_detector.inspect(align_res.aligned_roi)
                            heat_view = rule_res.overlay_bgr
                            self.gui_queue.put(("INSPECT_RULE_EVENT", (track_evt.object_id, rule_res)))
                            is_defect = rule_res.is_defect
                        else:
                            anom_res: AnomalyResult = self.detector.predict(align_res.aligned_roi)
                            heat_view = anom_res.overlay_bgr
                            self.gui_queue.put(("INSPECT_EVENT", (track_evt.object_id, anom_res)))
                            is_defect = anom_res.is_defect

                        if is_defect:
                            delay = self.vision_kick_delay_sec
                            obj_id = track_evt.object_id

                            def _delayed_kick_flow(kicker_num=1, target_id=obj_id):
                                if self.ctrl and self.running:
                                    self.ctrl.push_actuator(kicker_num)
                                self.gui_queue.put(("LOG", f"[KICKER 1] 웨이퍼 #{target_id} 배출 완료"))

                            threading.Timer(delay, _delayed_kick_flow).start()
                            self.gui_queue.put(("LOG", f"[KICKER 1] 웨이퍼 #{obj_id} 불량 배출 예약 ({delay:.2f}초 후 도달)"))
                    except Exception as e:
                        self.gui_queue.put(("LOG", f"[VISION ERROR] 검사 오류: {e}"))

                # Display tracking state overlay on RGB frame
                disp_frame = align_res.overlay_view.copy()
                cv2.putText(
                    disp_frame,
                    track_evt.status_text,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0) if track_evt.state in (TrackerState.INSPECTED, TrackerState.STOPPED_INSPECTING) else (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                self.gui_queue.put(("FRAME_RGB", disp_frame))
                if heat_view is not None:
                    self.gui_queue.put(("FRAME_HEAT", heat_view))

            except Exception as loop_e:
                self.gui_queue.put(("LOG", f"[VISION LOOP ERROR] {loop_e}"))
                time.sleep(0.05)

        cap.release()

    def _worker_depth(self):
        """HP60C 뎁스 카메라 검사 스레드."""
        shm_reader = ShmDepthReader()
        cam_reader = None
        inspector = None
        try:
            inspector = DepthWindowInspector.load_config(self.depth_cfg_path)
        except Exception:
            pass

        while self.running:
            time.sleep(0.03)
            rgb_f, depth_f = None, None

            if shm_reader.is_available():
                rgb_f, depth_f, _ = shm_reader.read()
            elif cam_reader is None:
                try:
                    cam_reader = HP60CCamera().open()
                except Exception:
                    pass

            if depth_f is None and cam_reader is not None:
                try:
                    depth_f = cam_reader.read()
                except Exception:
                    depth_f = None

            if depth_f is not None:
                depth_view = colorize_depth(depth_f)
                if inspector is not None:
                    res: InspectionResult = inspector.process_frame(depth_f, rgb_frame=rgb_f)
                    cx, cy = inspector.resolve_uv(depth_f.shape)
                    cv2.drawMarker(depth_view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                    cv2.circle(depth_view, (cx, cy), inspector.mask_radius_px, (0, 255, 0) if res.mask_ok else (200, 200, 200), 2)
                    if res.event == "OBJECT_DONE" and res.should_kick:
                        if self.ctrl:
                            self.ctrl.push_actuator(2)  # Depth Defect -> Kicker 2
                        self.gui_queue.put(("DEPTH_EVENT", res))

                self.gui_queue.put(("FRAME_DEPTH", depth_view))

        if cam_reader:
            cam_reader.close()
        shm_reader.close()

    # --- GUI 30FPS Loop & Event Processing ---

    def _update_gui(self):
        try:
            self.lbl_clock.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            while True:
                try:
                    evt_type, data = self.gui_queue.get_nowait()
                except Empty:
                    break

                try:
                    if evt_type == "FRAME_RGB":
                        self._render_image(self.canvas_rgb, data, (320, 220))
                    elif evt_type == "FRAME_HEAT":
                        self._render_image(self.canvas_heat, data, (320, 220))
                    elif evt_type == "FRAME_DEPTH":
                        self._render_image(self.canvas_depth, data, (320, 220))
                    elif evt_type == "INSPECT_EVENT":
                        obj_id, anom_res = data
                        self._handle_inspection_result(obj_id, anom_res)
                    elif evt_type == "INSPECT_RULE_EVENT":
                        obj_id, rule_res = data
                        self._handle_rule_inspection_result(obj_id, rule_res)
                    elif evt_type == "DEPTH_EVENT":
                        res: InspectionResult = data
                        self._log(f"[DEPTH NG] 높이 이상 ({res.min_depth_mm:.1f}mm) -> KICK 2")
                    elif evt_type == "LOG":
                        self._log(data)
                except Exception as evt_err:
                    print(f"[GUI EVENT ERROR] {evt_type}: {evt_err}")

            try:
                self._draw_trend_chart()
            except Exception:
                pass

            if hasattr(self, "canvas_wafer_map"):
                try:
                    map_img = self.spatial_analyzer.generate_wafer_map_image((220, 220))
                    self._render_image(self.canvas_wafer_map, map_img, (220, 220))
                except Exception as map_err:
                    print(f"[WAFER MAP RENDER ERROR] {map_err}")

        except Exception as main_loop_err:
            print(f"[GUI UPDATE LOOP ERROR] {main_loop_err}")
        finally:
            if self.running:
                self.root.after(33, self._update_gui)

    def _render_image(self, target_lbl: tk.Label, bgr_img: np.ndarray, size: Tuple[int, int]):
        if bgr_img is None:
            return
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        tw, th = size
        scale = min(tw / w, th / h)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized))
        target_lbl.config(image=img_tk)
        target_lbl.image = img_tk

    def _handle_rule_inspection_result(self, obj_id: int, res: RuleDefectResult):
        self.total_count += 1
        score = res.defect_score
        self.score_trend.pop(0)
        self.score_trend.append(score)

        defects_info = []
        for d in res.defects:
            area_val = float(getattr(d, "area", getattr(d, "area_px", 20.0)))
            center_val = getattr(d, "center", getattr(d, "center_xy", (128, 128)))
            cx = float(center_val[0]) if center_val else 128.0
            cy = float(center_val[1]) if center_val else 128.0
            defects_info.append({
                "r_mm": float(getattr(d, "polar_r_mm", 0.0)),
                "theta_deg": float(getattr(d, "polar_theta_deg", 0.0)),
                "defect_type": getattr(d, "defect_type", "MARKER_X"),
                "area_px": area_val,
                "cx_px": cx,
                "cy_px": cy,
            })

        self.spatial_analyzer.add_inspection_event(obj_id, res.is_defect, defects_info)

        polar_str = ""
        if res.defects:
            polar_str = "; ".join(
                [
                    f"{d.defect_type}({d.polar_r_mm:.0f}mm,{d.polar_theta_deg:.0f}°)"
                    for d in res.defects[:3]
                ]
            )

        severity = "CRITICAL" if res.is_defect and res.defect_count >= 2 else ("DEFECT" if res.is_defect else "NORMAL")
        pattern_str = res.defects[0].defect_type if res.defects else "NONE"

        rec = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object_id": obj_id,
            "verdict": "NG" if res.is_defect else "OK",
            "score": round(score, 1),
            "threshold": round(self.rule_detector.min_defect_area, 1),
            "severity": severity,
            "pattern": pattern_str,
            "defects": res.defect_count,
            "dispersion": 0.0,
            "polar": polar_str,
        }
        self.history_records.append(rec)

        if res.is_defect:
            self.ng_count += 1
            sev_color = "#EF4444" if severity == "CRITICAL" else "#F59E0B"
            self.lbl_severity.config(text=severity, bg=sev_color, fg="#FFFFFF")
            self.lbl_pattern.config(text=f"{pattern_str} (총 면적: {res.defect_score:.0f}px)", fg="#F87171")
            self._log(f"[RULE NG] #{obj_id} {res.description} -> Area={score:.0f}px")
        else:
            self.ok_count += 1
            self.lbl_severity.config(text="NORMAL (양품)", bg="#10B981", fg="#FFFFFF")
            self.lbl_pattern.config(text="NONE (정상 표면)", fg="#38BDF8")
            self._log(f"[RULE OK] #{obj_id} 양품 정상 통과 -> Area={score:.0f}px")

        self._update_stat_ui()
        self._update_spatial_ui()

    def _handle_inspection_result(self, obj_id: int, res: AnomalyResult):
        self.total_count += 1
        score = res.anomaly_score
        self.score_trend.pop(0)
        self.score_trend.append(score)

        defects_info = []
        if res.defects:
            for d in res.defects:
                pix = getattr(d, "centroid_pix", (128, 128))
                cx = float(pix[0]) if pix else 128.0
                cy = float(pix[1]) if pix else 128.0
                polar = getattr(d, "polar_coords", (0.0, 0.0))
                r_mm = float(polar[0]) if polar else 0.0
                theta_deg = float(polar[1]) if polar else 0.0
                defects_info.append({
                    "r_mm": r_mm,
                    "theta_deg": theta_deg,
                    "defect_type": "AI_ANOMALY",
                    "area_px": float(getattr(d, "area_px", 20.0)),
                    "cx_px": cx,
                    "cy_px": cy,
                })
        self.spatial_analyzer.add_inspection_event(obj_id, res.is_defect, defects_info)

        pat: PatternAnalysisResult = res.pattern or PatternAnalysisResult(
            severity="NORMAL",
            pattern_type="NONE",
            quadrant_density={"Q1_NE": 0.0, "Q2_SE": 0.0, "Q3_SW": 0.0, "Q4_NW": 0.0},
            defect_area_ratio=0.0,
            description="양품",
        )

        polar_str = ""
        if res.defects:
            polar_str = "; ".join([f"#{d.defect_id}({d.polar_coords[0]:.1f}mm,{d.polar_coords[1]:.0f}°)" for d in res.defects[:3]])

        rec = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object_id": obj_id,
            "verdict": "NG" if res.is_defect else "OK",
            "score": round(score, 3),
            "threshold": round(res.threshold, 3),
            "severity": pat.severity,
            "pattern": pat.pattern_type,
            "defects": res.num_defects,
            "dispersion": round(res.dispersion_mm, 2),
            "polar": polar_str,
        }
        self.history_records.append(rec)

        if res.is_defect:
            self.ng_count += 1
            sev_color = "#EF4444" if pat.severity == "CRITICAL" else "#F59E0B"
            self.lbl_severity.config(text=f"{pat.severity}", bg=sev_color, fg="#FFFFFF")
            self.lbl_pattern.config(text=f"{pat.pattern_type} ({pat.defect_area_ratio}%)", fg="#F87171")
            self._log(f"[VISION NG] #{obj_id} {pat.description} -> Score={score:.2f}")
        else:
            self.ok_count += 1
            self.lbl_severity.config(text="NORMAL (양품)", bg="#10B981", fg="#FFFFFF")
            self.lbl_pattern.config(text="NONE (정상 표면)", fg="#38BDF8")
            self._log(f"[VISION OK] #{obj_id} 양품 정상 통과 -> Score={score:.2f}")

        self._update_stat_ui()
        self._update_spatial_ui()

    def _draw_trend_chart(self):
        c = self.chart_canvas
        c.delete("all")
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw <= 10 or ch <= 10:
            return

        max_y = max(max(self.score_trend), self.current_threshold * 1.5, 10.0)
        th_y = ch - int((self.current_threshold / max_y) * (ch - 20)) - 10
        c.create_line(0, th_y, cw, th_y, fill="#EF4444", dash=(4, 4), width=1)
        c.create_text(cw - 50, th_y - 8, text=f"Threshold {self.current_threshold:.2f}", fill="#EF4444", font=("Helvetica", 7))

        n = len(self.score_trend)
        pts = []
        for i, val in enumerate(self.score_trend):
            x = int((i / (n - 1)) * (cw - 20)) + 10
            y = ch - int((val / max_y) * (ch - 20)) - 10
            pts.append((x, y))

        for i in range(len(pts) - 1):
            val = self.score_trend[i + 1]
            color = "#EF4444" if val >= self.current_threshold else "#10B981"
            c.create_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], fill=color, width=2)
            c.create_oval(pts[i + 1][0] - 2, pts[i + 1][1] - 2, pts[i + 1][0] + 2, pts[i + 1][1] + 2, fill=color, outline="")

    def _on_close(self):
        self.running = False
        if self.ctrl:
            self.ctrl.system_stop()
            self.ctrl.close()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smart Factory GUI Dashboard")
    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino serial port (e.g. /dev/ttyACM0)")
    parser.add_argument("--cam-id", default=0, type=int, help="Camera ID (default 0)")
    parser.add_argument("--depth-cfg", default="depth_cfg.json", type=str, help="Depth config JSON")
    parser.add_argument("--weights", default="weights/patchcore_resnet18.pkl", type=str, help="Model weights")
    args = parser.parse_args()

    root = tk.Tk()
    app = SmartFactoryDashboard(
        root=root,
        device_port=args.device,
        cam_id=args.cam_id,
        depth_cfg=args.depth_cfg,
        weights_path=args.weights,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
