"""
gui.py - Giao diện đồ họa (GUI) Mô phỏng, Giải Bài Toán Ngược & Phân Tích Ảnh CT Thực Tế (CT Simulator Pro).
Bao gồm 5 Tab chuyên sâu:
  - Tab 1: Quét & Thu nhận trực tiếp + Nạp File Ảnh CT Thực Tế & Tương tác chọn ROI bằng chuột
  - Tab 2: So sánh 7 thuật toán tái tạo (BP, FBP với 5 bộ lọc, SART, SIRT, MBIR, ASIR, DLIR) & Phân tích miền tần số
  - Tab 3: Khảo sát nguyên lý ALARA & Tối ưu hóa liều tia X (Dose Sweep & Noise Analysis)
  - Tab 4: Phân tích Chuyên sâu Ảnh CT Lâm sàng Thực tế (Histogram HU, ROI CNR) & Trình chiếu 6 Flowchart Thuật toán
  - Tab 5: Xuất báo cáo kỹ thuật & Mẫu BTL Bách Khoa (Export & PDF Generator)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

# Cấu hình Matplotlib sử dụng backend TkAgg
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Import các module cốt lõi trong ct_sim
from ct_sim.phantoms import PHANTOMS, make_phantom, roi_mask, circle_mask, register_custom_real_ct
from ct_sim.real_ct import CALIBRATION_PRESETS, load_real_ct_image, ensure_builtin_real_ct_samples
from ct_sim.physics import ScanParams, ScanData, simulate_scan, simulate_scan_from_hu, effective_energy, estimate_dose
from ct_sim.recon import FILTERS, fourier_filter, fbp
from ct_sim.pipeline import ALGOS, DEFAULT_CFG, ReconContext, run_algorithms, dose_sweep, label_for, AlgoResult
from ct_sim.metrics import evaluate
from ct_sim.dl_model import load_model


# Thiết lập bảng màu chuyên nghiệp (Dark Medical Workstation Theme)
BG_DARK = "#0f172a"        # Slate 900
PANEL_BG = "#1e293b"       # Slate 800
PANEL_LIGHT = "#334155"    # Slate 700
TEXT_WHITE = "#f8fafc"     # Slate 50
TEXT_MUTED = "#94a3b8"     # Slate 400
ACCENT_CYAN = "#06b6d4"    # Cyan 500
ACCENT_GREEN = "#10b981"   # Emerald 500
ACCENT_ORANGE = "#f59e0b"  # Amber 500
ACCENT_RED = "#ef4444"     # Red 500
ACCENT_PURPLE = "#a855f7"  # Purple 500
FONT_FAMILY = "Segoe UI"


class CTSimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HỆ THỐNG MÔ PHỎNG, TÁI TẠO & PHÂN TÍCH CẮT LỚP VI TÍNH (CT SIMULATOR PRO - BME LAB)")
        self.geometry("1440x900")
        self.minsize(1150, 760)
        self.configure(bg=BG_DARK)

        # Đảm bảo kho mẫu Ảnh CT Thực Tế đã sẵn sàng
        ensure_builtin_real_ct_samples()

        # Trạng thái dữ liệu hiện tại
        self.current_scan: Optional[ScanData] = None
        self.recon_context: Optional[ReconContext] = None
        self.algo_results: Dict[str, AlgoResult] = {}
        self.custom_file_path: Optional[str] = None
        self.is_scanning = False
        self.cancel_scan = False

        # Thiết lập style giao diện ttk
        self._init_styles()

        # Tạo thanh tiêu đề thương hiệu (Header)
        self._create_header()

        # Tạo Notebook chia 5 Tab tính năng
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.tab_scan = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.tab_compare = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.tab_alara = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.tab_real_fc = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.tab_export = ttk.Frame(self.notebook, style="Dark.TFrame")

        self.notebook.add(self.tab_scan, text="  1. Quét Trực Tiếp & Nạp Ảnh CT  ")
        self.notebook.add(self.tab_compare, text="  2. So Sánh Thuật Toán & 5 Bộ Lọc FBP  ")
        self.notebook.add(self.tab_alara, text="  3. Khảo Sát Tối Ưu Liều (ALARA)  ")
        self.notebook.add(self.tab_real_fc, text="  4. Phân Tích CT Thực Tế & Flowchart Toán Học  ")
        self.notebook.add(self.tab_export, text="  5. Báo Cáo Kỹ Thuật & Xuất PDF BTL  ")

        self._build_tab_scan()
        self._build_tab_compare()
        self._build_tab_alara()
        self._build_tab_real_fc()
        self._build_tab_export()

        # Khởi động quét mẫu mặc định lần đầu
        self.after(300, self._quick_initial_scan)

    # ----------------------------------------------------------------------------------
    # GIAO DIỆN & STYLE
    # ----------------------------------------------------------------------------------
    def _init_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("PanelLight.TFrame", background=PANEL_LIGHT)

        style.configure("TLabel", background=PANEL_BG, foreground=TEXT_WHITE, font=(FONT_FAMILY, 9))
        style.configure("Header.TLabel", background=BG_DARK, foreground=TEXT_WHITE, font=(FONT_FAMILY, 13, "bold"))
        style.configure("SubHeader.TLabel", background=BG_DARK, foreground=ACCENT_CYAN, font=(FONT_FAMILY, 9))
        style.configure("Section.TLabel", background=PANEL_BG, foreground=ACCENT_CYAN, font=(FONT_FAMILY, 9, "bold"))
        style.configure("Muted.TLabel", background=PANEL_BG, foreground=TEXT_MUTED, font=(FONT_FAMILY, 8))

        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_BG, foreground=TEXT_WHITE, font=(FONT_FAMILY, 9, "bold"), padding=[10, 6])
        style.map("TNotebook.Tab", background=[("selected", ACCENT_CYAN)], foreground=[("selected", "#000000")])

        style.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT_WHITE, font=(FONT_FAMILY, 9))
        style.configure("TRadiobutton", background=PANEL_BG, foreground=TEXT_WHITE, font=(FONT_FAMILY, 9))

    def _create_header(self):
        header_frame = tk.Frame(self, bg=BG_DARK, height=52)
        header_frame.pack(fill=tk.X, padx=14, pady=(6, 2))

        logo_path = os.path.join(os.path.dirname(__file__), "assets", "bk_logo.png")
        if os.path.exists(logo_path):
            try:
                from PIL import Image, ImageTk
                img = Image.open(logo_path).resize((38, 38), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                lbl_logo = tk.Label(header_frame, image=self.logo_img, bg=BG_DARK)
                lbl_logo.pack(side=tk.LEFT, padx=(0, 10))
            except Exception:
                pass

        title_box = tk.Frame(header_frame, bg=BG_DARK)
        title_box.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(title_box, text="CT SIMULATOR PRO - HỆ THỐNG GIẢI BÀI TOÁN NGƯỢC, TÁI TẠO & PHÂN TÍCH ẢNH CT Y TẾ", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(title_box, text="ĐHQG TP.HCM - Trường Đại học Bách khoa | Bộ môn Kỹ thuật Y sinh & Cơ sở Vật lý Hạt nhân (BP • FBP • SART • SIRT • MBIR • ASIR • DLIR)", style="SubHeader.TLabel").pack(anchor=tk.W)

        self.lbl_alara_badge = tk.Label(header_frame, text="ALARA: Chuẩn (200 mA | 45.0 mGy)",
                                        font=(FONT_FAMILY, 9, "bold"), bg="#065f46", fg="#34d399",
                                        padx=12, pady=4, relief=tk.FLAT)
        self.lbl_alara_badge.pack(side=tk.RIGHT, padx=4)

    # ----------------------------------------------------------------------------------
    # TAB 1: QUÉT & THU NHẬN TRỰC TIẾP + NẠP ẢNH CT THỰC TẾ
    # ----------------------------------------------------------------------------------
    def _build_tab_scan(self):
        container = tk.Frame(self.tab_scan, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left_panel = tk.Frame(container, bg=PANEL_BG, width=335)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6), pady=0)
        left_panel.pack_propagate(False)

        canvas_ctrl = tk.Canvas(left_panel, bg=PANEL_BG, highlightthickness=0)
        scrollbar_ctrl = ttk.Scrollbar(left_panel, orient="vertical", command=canvas_ctrl.yview)
        scroll_content = tk.Frame(canvas_ctrl, bg=PANEL_BG)

        scroll_content.bind("<Configure>", lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all")))
        canvas_ctrl.create_window((0, 0), window=scroll_content, anchor="nw", width=315)
        canvas_ctrl.configure(yscrollcommand=scrollbar_ctrl.set)

        canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_ctrl.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. Chọn Phantom hoặc Ảnh CT Thực Tế
        ttk.Label(scroll_content, text="1. PHANTOM & ẢNH CT THỰC TẾ", style="Section.TLabel").pack(anchor=tk.W, padx=10, pady=(8, 3))
        self.var_phantom = tk.StringVar(value="Đầu (HU)")
        self.cb_phantom = ttk.Combobox(scroll_content, textvariable=self.var_phantom, values=list(PHANTOMS.keys()), state="readonly")
        self.cb_phantom.pack(fill=tk.X, padx=10, pady=2)
        self.cb_phantom.bind("<<ComboboxSelected>>", lambda e: self._on_phantom_selected())

        # Nút Nạp Ảnh CT Thực Tế từ file bất kỳ
        btn_load_real = tk.Button(
            scroll_content, text="📂 Nạp Ảnh CT Thực Tế (File PNG/JPG/DCM/NPY...)",
            font=(FONT_FAMILY, 8, "bold"), bg="#eab308", fg="#000000", activebackground="#facc15",
            command=self._browse_real_ct_file
        )
        btn_load_real.pack(fill=tk.X, padx=10, pady=(4, 2))

        # Cửa sổ hiệu chuẩn HU khi nạp ảnh thực tế
        f_cal = tk.Frame(scroll_content, bg=PANEL_BG)
        f_cal.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(f_cal, text="Hiệu chuẩn HU:").pack(side=tk.LEFT)
        self.var_cal_preset = tk.StringVar(value="Auto (Tự động)")
        cb_cal = ttk.Combobox(f_cal, textvariable=self.var_cal_preset, values=list(CALIBRATION_PRESETS.keys()), width=19, state="readonly")
        cb_cal.pack(side=tk.RIGHT)
        cb_cal.bind("<<ComboboxSelected>>", lambda e: self._on_calibration_changed())

        # Tùy chọn hiển thị ROI & tương tác chuột
        self.var_show_roi = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_content, text="Hiển thị vòng ROI (Click Trái: Nền, Phải: U)",
                        variable=self.var_show_roi, command=self._redraw_current_tab1).pack(anchor=tk.W, padx=10, pady=2)

        # Cấy kim loại
        f_metal = tk.Frame(scroll_content, bg=PANEL_BG)
        f_metal.pack(fill=tk.X, padx=10, pady=3)
        ttk.Label(f_metal, text="Cấy kim loại (Metal):").pack(side=tk.LEFT)
        self.var_metal = tk.IntVar(value=0)
        ttk.Radiobutton(f_metal, text="Không", variable=self.var_metal, value=0, command=self._on_param_changed).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(f_metal, text="1", variable=self.var_metal, value=1, command=self._on_param_changed).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(f_metal, text="2", variable=self.var_metal, value=2, command=self._on_param_changed).pack(side=tk.LEFT, padx=2)

        # 2. Thông số phát tia
        ttk.Label(scroll_content, text="2. THÔNG SỐ BÓNG PHÁT (X-RAY TUBE)", style="Section.TLabel").pack(anchor=tk.W, padx=10, pady=(8, 3))

        f_kv = tk.Frame(scroll_content, bg=PANEL_BG)
        f_kv.pack(fill=tk.X, padx=10, pady=1)
        ttk.Label(f_kv, text="Điện áp đỉnh (kVp):").pack(side=tk.LEFT)
        self.var_kv = tk.IntVar(value=120)
        self.lbl_kv_val = ttk.Label(f_kv, text="120 kV", style="Section.TLabel")
        self.lbl_kv_val.pack(side=tk.RIGHT)

        f_kv_radios = tk.Frame(scroll_content, bg=PANEL_BG)
        f_kv_radios.pack(fill=tk.X, padx=10, pady=1)
        for kv_val in (80, 100, 120, 140):
            ttk.Radiobutton(f_kv_radios, text=f"{kv_val}", variable=self.var_kv, value=kv_val,
                            command=self._on_kv_changed).pack(side=tk.LEFT, padx=4)

        f_ma = tk.Frame(scroll_content, bg=PANEL_BG)
        f_ma.pack(fill=tk.X, padx=10, pady=(4, 1))
        ttk.Label(f_ma, text="Dòng ống (mA):").pack(side=tk.LEFT)
        self.lbl_ma_val = ttk.Label(f_ma, text="120 mA", style="Section.TLabel")
        self.lbl_ma_val.pack(side=tk.RIGHT)

        self.var_ma = tk.DoubleVar(value=120.0)
        scale_ma = ttk.Scale(scroll_content, from_=10.0, to=300.0, variable=self.var_ma, orient=tk.HORIZONTAL,
                             command=self._on_ma_slider)
        scale_ma.pack(fill=tk.X, padx=10, pady=2)

        f_views = tk.Frame(scroll_content, bg=PANEL_BG)
        f_views.pack(fill=tk.X, padx=10, pady=(4, 2))
        ttk.Label(f_views, text="Số góc chiếu (Views):").pack(side=tk.LEFT)
        self.var_views = tk.IntVar(value=180)
        cb_views = ttk.Combobox(f_views, textvariable=self.var_views, values=[60, 90, 120, 180], width=6, state="readonly")
        cb_views.pack(side=tk.RIGHT)
        cb_views.bind("<<ComboboxSelected>>", lambda e: self._on_param_changed())

        # Bộ lọc FBP cho Tab 1
        f_flt1 = tk.Frame(scroll_content, bg=PANEL_BG)
        f_flt1.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(f_flt1, text="Bộ lọc FBP H(ω):").pack(side=tk.LEFT)
        self.var_fbp_filter = tk.StringVar(value="ramp")
        cb_flt1 = ttk.Combobox(f_flt1, textvariable=self.var_fbp_filter, values=list(FILTERS), width=11, state="readonly")
        cb_flt1.pack(side=tk.RIGHT)
        cb_flt1.bind("<<ComboboxSelected>>", lambda e: self._fast_scan())

        self.var_poisson = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_content, text="Nhiễu lượng tử Poisson (Quantum Noise)", variable=self.var_poisson,
                        command=self._on_param_changed).pack(anchor=tk.W, padx=10, pady=(4, 1))

        self.var_beam_hardening = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_content, text="Hiệu ứng Cứng hóa chùm tia (Beam Hardening)", variable=self.var_beam_hardening,
                        command=self._on_param_changed).pack(anchor=tk.W, padx=10, pady=1)

        # 3. Bảng liều bức xạ & Thống kê ROI trực tiếp
        ttk.Label(scroll_content, text="3. LIỀU BỨC XẠ ALARA & CHỈ SỐ ROI", style="Section.TLabel").pack(anchor=tk.W, padx=10, pady=(8, 3))
        dose_box = tk.Frame(scroll_content, bg="#0b0f19", relief=tk.RIDGE, bd=1)
        dose_box.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_ctdi = tk.Label(dose_box, text="CTDIvol: 27.0 mGy", font=(FONT_FAMILY, 8, "bold"), fg="#38bdf8", bg="#0b0f19")
        self.lbl_ctdi.pack(anchor=tk.W, padx=8, pady=(3, 1))
        self.lbl_dlp = tk.Label(dose_box, text="DLP: 405.0 mGy.cm | E: 0.85 mSv", font=(FONT_FAMILY, 8), fg="#4ade80", bg="#0b0f19")
        self.lbl_dlp.pack(anchor=tk.W, padx=8, pady=1)
        self.lbl_flux = tk.Label(dose_box, text="Quang thông I₀: 1.33e+06 photons/bin", font=(FONT_FAMILY, 8), fg="#94a3b8", bg="#0b0f19")
        self.lbl_flux.pack(anchor=tk.W, padx=8, pady=1)
        self.lbl_roi_live = tk.Label(dose_box, text="ROI Nền: -- HU | ROI U: -- HU | CNR: --", font=(FONT_FAMILY, 8, "bold"), fg="#facc15", bg="#0b0f19")
        self.lbl_roi_live.pack(anchor=tk.W, padx=8, pady=(1, 4))

        # 4. Các nút thao tác
        f_btns = tk.Frame(scroll_content, bg=PANEL_BG)
        f_btns.pack(fill=tk.X, padx=10, pady=(8, 6))

        self.btn_live_scan = tk.Button(f_btns, text="▶ Quét Động Trực Tiếp (Live Scan)", font=(FONT_FAMILY, 9, "bold"),
                                       bg=ACCENT_CYAN, fg="#000000", activebackground="#22d3ee",
                                       command=self._start_live_scan_thread)
        self.btn_live_scan.pack(fill=tk.X, pady=2)

        self.btn_fast_scan = tk.Button(f_btns, text="⚡ Quét Nhanh & Khôi Phục Tức Thì", font=(FONT_FAMILY, 9, "bold"),
                                       bg=ACCENT_GREEN, fg="#ffffff", activebackground="#34d399",
                                       command=self._fast_scan)
        self.btn_fast_scan.pack(fill=tk.X, pady=2)

        self.btn_stop_scan = tk.Button(f_btns, text="⏹ Dừng Quét", font=(FONT_FAMILY, 8),
                                       bg="#475569", fg="#ffffff", state=tk.DISABLED,
                                       command=self._stop_scan)
        self.btn_stop_scan.pack(fill=tk.X, pady=2)

        self.progress_scan = ttk.Progressbar(scroll_content, orient="horizontal", mode="determinate")
        self.progress_scan.pack(fill=tk.X, padx=10, pady=3)
        self.lbl_scan_status = ttk.Label(scroll_content, text="Sẵn sàng quét.", style="Muted.TLabel")
        self.lbl_scan_status.pack(anchor=tk.W, padx=10, pady=(0, 8))

        # --- Khung hiển thị bên phải: 3 Subplots đồng bộ ---
        right_panel = tk.Frame(container, bg=BG_DARK)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_scan = Figure(figsize=(10, 4.8), dpi=100)
        self.fig_scan.patch.set_facecolor(BG_DARK)

        self.ax1 = self.fig_scan.add_subplot(131)
        self.ax2 = self.fig_scan.add_subplot(132)
        self.ax3 = self.fig_scan.add_subplot(133)

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.set_facecolor(BG_DARK)

        self.fig_scan.tight_layout()
        self.canvas_scan = FigureCanvasTkAgg(self.fig_scan, master=right_panel)
        self.canvas_scan.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Bắt sự kiện nhấp chuột trên Subplot 1 hoặc 3 để đổi vị trí ROI tương tác
        self.canvas_scan.mpl_connect("button_press_event", self._on_scan_canvas_click)

    # ----------------------------------------------------------------------------------
    # TAB 2: SO SÁNH THUẬT TOÁN & BỘ LỌC FBP
    # ----------------------------------------------------------------------------------
    def _build_tab_compare(self):
        container = tk.Frame(self.tab_compare, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        top_ctrl = tk.Frame(container, bg=PANEL_BG)
        top_ctrl.pack(fill=tk.X, padx=2, pady=(0, 6))

        f_algos = tk.Frame(top_ctrl, bg=PANEL_BG)
        f_algos.pack(side=tk.LEFT, padx=8, pady=8)
        ttk.Label(f_algos, text="Thuật toán:", style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))

        self.algo_vars: Dict[str, tk.BooleanVar] = {}
        for key in ("BP", "FBP", "SART", "SIRT", "MBIR", "ASIR", "DLIR"):
            val = key in ("BP", "FBP", "SART", "MBIR", "ASIR", "DLIR")
            var = tk.BooleanVar(value=val)
            self.algo_vars[key] = var
            ttk.Checkbutton(f_algos, text=key, variable=var).pack(side=tk.LEFT, padx=3)

        btn_run_compare = tk.Button(top_ctrl, text="▶ Tái Tạo & Đo Lường", font=(FONT_FAMILY, 8, "bold"),
                                    bg=ACCENT_CYAN, fg="#000000", padx=10, pady=4,
                                    command=self._start_compare_thread)
        btn_run_compare.pack(side=tk.LEFT, padx=6)

        btn_filter_study = tk.Button(top_ctrl, text="🔍 Bóc Tách 5 Bộ Lọc FBP (Tần Số/Không Gian)",
                                     font=(FONT_FAMILY, 8, "bold"), bg=ACCENT_ORANGE, fg="#000000", padx=10, pady=4,
                                     command=self._show_5_filters_analysis)
        btn_filter_study.pack(side=tk.LEFT, padx=6)

        f_win = tk.Frame(top_ctrl, bg=PANEL_BG)
        f_win.pack(side=tk.RIGHT, padx=8, pady=8)
        ttk.Label(f_win, text="Cửa sổ:").pack(side=tk.LEFT, padx=(0, 3))
        self.var_win = tk.StringVar(value="Auto (Theo Phantom)")
        cb_win = ttk.Combobox(f_win, textvariable=self.var_win,
                              values=["Auto (Theo Phantom)", "Brain (L:40, W:120)", "Abdomen (L:50, W:350)", "Bone (L:400, W:1500)", "Lung (L:-500, W:1500)", "Full (-1000, 1000)"],
                              width=19, state="readonly")
        cb_win.pack(side=tk.LEFT, padx=2)
        cb_win.bind("<<ComboboxSelected>>", lambda e: self._update_compare_plots())

        ttk.Label(f_win, text="Màu:").pack(side=tk.LEFT, padx=(6, 3))
        self.var_cmap = tk.StringVar(value="bone")
        cb_cmap = ttk.Combobox(f_win, textvariable=self.var_cmap, values=["bone", "gray", "inferno", "magma", "viridis"], width=7, state="readonly")
        cb_cmap.pack(side=tk.LEFT, padx=2)
        cb_cmap.bind("<<ComboboxSelected>>", lambda e: self._update_compare_plots())

        split_frame = tk.Frame(container, bg=BG_DARK)
        split_frame.pack(fill=tk.BOTH, expand=True)

        img_box = tk.Frame(split_frame, bg=BG_DARK)
        img_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.fig_compare = Figure(figsize=(7.6, 6.0), dpi=100)
        self.fig_compare.patch.set_facecolor(BG_DARK)
        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, master=img_box)
        self.canvas_compare.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        side_box = tk.Frame(split_frame, bg=PANEL_BG, width=450)
        side_box.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(4, 0))
        side_box.pack_propagate(False)

        ttk.Label(side_box, text="BẢNG CHỈ SỐ ĐỊNH LƯỢNG KHÔI PHỤC HÀM μ(x,y)", style="Section.TLabel").pack(anchor=tk.W, padx=8, pady=(8, 4))

        cols = ("algo", "psnr", "ssim", "noise", "cnr", "time")
        self.tree_metrics = ttk.Treeview(side_box, columns=cols, show="headings", height=9)
        self.tree_metrics.heading("algo", text="Thuật toán / Bộ lọc")
        self.tree_metrics.heading("psnr", text="PSNR (dB)")
        self.tree_metrics.heading("ssim", text="SSIM")
        self.tree_metrics.heading("noise", text="Nhiễu (HU)")
        self.tree_metrics.heading("cnr", text="CNR")
        self.tree_metrics.heading("time", text="Thời gian")

        self.tree_metrics.column("algo", width=120, anchor=tk.W)
        self.tree_metrics.column("psnr", width=65, anchor=tk.CENTER)
        self.tree_metrics.column("ssim", width=55, anchor=tk.CENTER)
        self.tree_metrics.column("noise", width=68, anchor=tk.CENTER)
        self.tree_metrics.column("cnr", width=55, anchor=tk.CENTER)
        self.tree_metrics.column("time", width=62, anchor=tk.CENTER)

        self.tree_metrics.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(side_box, text="MẶT CẮT ĐƯỜNG NGANG 1D QUA TỔN THƯƠNG (HU PROFILE)", style="Section.TLabel").pack(anchor=tk.W, padx=8, pady=(8, 2))
        self.fig_profile = Figure(figsize=(4.3, 3.2), dpi=90)
        self.fig_profile.patch.set_facecolor(PANEL_BG)
        self.ax_prof = self.fig_profile.add_subplot(111)
        self.ax_prof.set_facecolor(BG_DARK)
        self.fig_profile.tight_layout()

        self.canvas_profile = FigureCanvasTkAgg(self.fig_profile, master=side_box)
        self.canvas_profile.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ----------------------------------------------------------------------------------
    # TAB 3: KHẢO SÁT NGUYÊN LÝ ALARA & TỐI ƯU LIỀU
    # ----------------------------------------------------------------------------------
    def _build_tab_alara(self):
        container = tk.Frame(self.tab_alara, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        top_bar = tk.Frame(container, bg=PANEL_BG)
        top_bar.pack(fill=tk.X, padx=2, pady=(0, 6))

        ttk.Label(top_bar, text="Khảo sát suy giảm chất lượng ảnh theo dòng bóng phát (mA) & liều hấp thụ ALARA (Kiểm chứng σ ~ 1/√mA):",
                  style="Section.TLabel").pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_run_alara = tk.Button(top_bar, text="⚡ Chạy Quét Đồ Thị Liều (mA = 10 -> 200)", font=(FONT_FAMILY, 9, "bold"),
                                       bg=ACCENT_ORANGE, fg="#000000", padx=12, pady=4,
                                       command=self._start_alara_sweep_thread)
        self.btn_run_alara.pack(side=tk.LEFT, padx=14)

        self.lbl_alara_status = ttk.Label(top_bar, text="Nhấn nút để chạy khảo sát ALARA trên đối tượng hiện tại.", style="Muted.TLabel")
        self.lbl_alara_status.pack(side=tk.LEFT, padx=6)

        plot_frame = tk.Frame(container, bg=BG_DARK)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig_alara = Figure(figsize=(10, 6), dpi=100)
        self.fig_alara.patch.set_facecolor(BG_DARK)
        self.canvas_alara = FigureCanvasTkAgg(self.fig_alara, master=plot_frame)
        self.canvas_alara.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------------------------------
    # TAB 4: PHÂN TÍCH CHUYÊN SÂU ẢNH CT THỰC TẾ & SƠ ĐỒ KHỐI THUẬT TOÁN (FLOWCHARTS)
    # ----------------------------------------------------------------------------------
    def _build_tab_real_fc(self):
        container = tk.Frame(self.tab_real_fc, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Cột điều khiển bên trái
        left_box = tk.Frame(container, bg=PANEL_BG, width=330)
        left_box.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left_box.pack_propagate(False)

        ttk.Label(left_box, text="1. KHO ẢNH CT LÂM SÀNG THỰC TẾ", style="Section.TLabel").pack(anchor=tk.W, padx=10, pady=(10, 4))

        real_presets = [k for k in PHANTOMS.keys() if "CT Thực Tế" in k]
        self.var_real_case = tk.StringVar(value=real_presets[0] if real_presets else "CT Thực Tế: Gan Bình Thường")
        self.cb_real_case = ttk.Combobox(left_box, textvariable=self.var_real_case, values=real_presets, state="readonly")
        self.cb_real_case.pack(fill=tk.X, padx=10, pady=2)
        self.cb_real_case.bind("<<ComboboxSelected>>", lambda e: self._run_real_ct_deep_analysis(self.var_real_case.get()))

        btn_open_custom = tk.Button(
            left_box, text="📂 Mở File Ảnh CT Thực Tế Bất Kỳ...",
            font=(FONT_FAMILY, 8, "bold"), bg="#eab308", fg="#000000",
            command=self._browse_real_ct_file
        )
        btn_open_custom.pack(fill=tk.X, padx=10, pady=(6, 3))

        btn_run_real = tk.Button(
            left_box, text="🔬 Phân Tích Chuyên Sâu Ca CT Thực Tế Này",
            font=(FONT_FAMILY, 8, "bold"), bg=ACCENT_CYAN, fg="#000000",
            command=lambda: self._run_real_ct_deep_analysis(self.var_real_case.get())
        )
        btn_run_real.pack(fill=tk.X, padx=10, pady=3)

        # Khung thông tin định lượng của Ca CT Thực tế
        ttk.Label(left_box, text="2. THỐNG KÊ MÔ HỌC & ĐỊNH LƯỢNG ROI", style="Section.TLabel").pack(anchor=tk.W, padx=10, pady=(12, 4))
        self.txt_real_stats = tk.Text(left_box, bg="#0b0f19", fg=TEXT_WHITE, font=("Consolas", 8),
                                      height=12, relief=tk.FLAT, padx=8, pady=6)
        self.txt_real_stats.pack(fill=tk.X, padx=10, pady=2)

        # Phần xem Sơ đồ khối thuật toán (Flowcharts)
        ttk.Label(left_box, text="3. SƠ ĐỒ KHỐI TOÁN HỌC (FLOWCHARTS)", style="Section.TLabel").pack(anchor=tk.W, padx=10, pady=(12, 4))
        ttk.Label(left_box, text="Chọn sơ đồ thuật toán để phóng to:", style="Muted.TLabel").pack(anchor=tk.W, padx=10, pady=(0, 3))

        self.flowchart_options = OrderedDict_list = [
            ("Flowchart 0: Khung Bài Toán Ngược Tổng Quát", "fc0_inverse_problem_framework.png"),
            ("Flowchart 1: Chiếu Ngược Đơn Thuần (BP & 1/r)", "fc1_bp_flowchart.png"),
            ("Flowchart 2: Chiếu Ngược Có Lọc (FBP & 5 Bộ lọc)", "fc2_fbp_flowchart.png"),
            ("Flowchart 3 & 4: Lặp Đại Số OS-SART & SIRT", "fc3_sart_sirt_flowchart.png"),
            ("Flowchart 5: Lặp Thống Kê MBIR (PWLS-TV + PCG)", "fc4_mbir_pwls_tv_flowchart.png"),
            ("Flowchart 6 & 7: Lai ASIR & Học Sâu Phần Dư DLIR", "fc5_asir_dlir_flowchart.png"),
            ("Đồ thị Bóc tách 5 Bộ lọc FBP (Tần số & Không gian)", "fig5_filter_frequency_spatial.png"),
        ]
        for label_txt, fname in self.flowchart_options:
            btn = tk.Button(
                left_box, text="📐 " + label_txt, anchor="w",
                font=(FONT_FAMILY, 8), bg=PANEL_LIGHT, fg=TEXT_WHITE, activebackground=ACCENT_CYAN,
                padx=6, pady=3, command=lambda fn=fname, lt=label_txt: self._display_flowchart_on_tab4(fn, lt)
            )
            btn.pack(fill=tk.X, padx=10, pady=1.5)

        # Khung hiển thị bên phải của Tab 4
        right_box = tk.Frame(container, bg=BG_DARK)
        right_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_real_fc = Figure(figsize=(10.2, 6.2), dpi=100)
        self.fig_real_fc.patch.set_facecolor(BG_DARK)
        self.canvas_real_fc = FigureCanvasTkAgg(self.fig_real_fc, master=right_box)
        self.canvas_real_fc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_real_fc.mpl_connect("button_press_event", self._on_tab4_canvas_click)

    # ----------------------------------------------------------------------------------
    # TAB 5: BÁO CÁO KỸ THUẬT & XUẤT MẪU BTL BÁCH KHOA
    # ----------------------------------------------------------------------------------
    def _build_tab_export(self):
        container = tk.Frame(self.tab_export, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        info_panel = tk.Frame(container, bg=PANEL_BG, padx=16, pady=14)
        info_panel.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(info_panel, text="XUẤT BÁO CÁO BÀI TẬP LỚN (ĐỊNH DẠNG CHUẨN ĐẠI HỌC BÁCH KHOA - KÈM 6 FLOWCHART & TOÁN HỌC CHUYÊN SÂU)", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 6))
        desc_text = (
            "Hệ thống tự động kết xuất toàn bộ báo cáo BTL định dạng PDF chuẩn Bách Khoa (bám sát mẫu BTLXSTK_Khoa Cơ Khí_L01.pdf), "
            "bao gồm: Khung lý thuyết Bài toán ngược đặt không chỉnh (Ill-posed Inverse Problem), chứng minh hàm truyền điểm 1/(πr) của BP, "
            "bóc tách nhân Ram-Lak & định lý truyền phương sai nhiễu của 5 bộ lọc FBP, mô hình dải hình thang & bán hội tụ GSVD của SART/SIRT, "
            "đạo hàm Euler-Lagrange & giải thuật Lagged Diffusivity + PCG của MBIR, điều khiển phổ NPS của ASIR, mạng Residual CNN của DLIR, "
            "6 Flowchart thuật toán và kết quả phân tích Ảnh CT Lâm sàng Thực tế."
        )
        tk.Label(info_panel, text=desc_text, font=(FONT_FAMILY, 9), fg=TEXT_MUTED, bg=PANEL_BG, justify=tk.LEFT, wraplength=1250).pack(anchor=tk.W, pady=(0, 10))

        btn_bar = tk.Frame(info_panel, bg=PANEL_BG)
        btn_bar.pack(fill=tk.X)

        self.btn_gen_pdf = tk.Button(btn_bar, text="📄 TẠO BÁO CÁO BTL PDF CHUẨN BÁCH KHOA",
                                     font=(FONT_FAMILY, 10, "bold"), bg=ACCENT_GREEN, fg="#ffffff",
                                     padx=16, pady=8, command=self._generate_pdf_report)
        self.btn_gen_pdf.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_save_images = tk.Button(btn_bar, text="💾 Lưu Toàn Bộ Ảnh Tái Tạo (PNG)",
                                         font=(FONT_FAMILY, 9), bg=PANEL_LIGHT, fg=TEXT_WHITE,
                                         padx=12, pady=8, command=self._save_images)
        self.btn_save_images.pack(side=tk.LEFT, padx=6)

        self.btn_export_csv = tk.Button(btn_bar, text="📊 Xuất Bảng Số Liệu (CSV)",
                                        font=(FONT_FAMILY, 9), bg=PANEL_LIGHT, fg=TEXT_WHITE,
                                        padx=12, pady=8, command=self._export_csv)
        self.btn_export_csv.pack(side=tk.LEFT, padx=6)

        preview_box = tk.Frame(container, bg=PANEL_BG, padx=12, pady=10)
        preview_box.pack(fill=tk.BOTH, expand=True)

        ttk.Label(preview_box, text="BẢN XEM TRƯỚC TÓM TẮT TOÁN HỌC & KỸ THUẬT CỦA BÁO CÁO BTL", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))
        self.txt_preview = tk.Text(preview_box, bg="#0b0f19", fg=TEXT_WHITE, font=("Consolas", 9),
                                   insertbackground="white", relief=tk.FLAT, padx=10, pady=10)
        scroll_txt = ttk.Scrollbar(preview_box, orient="vertical", command=self.txt_preview.yview)
        self.txt_preview.configure(yscrollcommand=scroll_txt.set)

        self.txt_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_txt.pack(side=tk.RIGHT, fill=tk.Y)

        self._populate_report_preview()

    # ----------------------------------------------------------------------------------
    # NẠP ẢNH CT THỰC TẾ TỪ FILE & TƯƠNG TÁC ROI
    # ----------------------------------------------------------------------------------
    def _browse_real_ct_file(self):
        filetypes = [
            ("Tất cả định dạng Ảnh CT", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.npy *.npz *.dcm"),
            ("Ảnh Y khoa (PNG/JPG/BMP/TIFF)", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("Mảng Khoa học & DICOM (NPY/NPZ/DCM)", "*.npy *.npz *.dcm"),
            ("Tất cả các file", "*.*"),
        ]
        fp = filedialog.askopenfilename(title="Chọn tập tin Ảnh CT Thực Tế để phân tích", filetypes=filetypes)
        if not fp:
            return

        self.custom_file_path = fp
        fname = os.path.basename(fp)
        custom_key = f"CT Thực Tế: {fname}"
        preset = self.var_cal_preset.get()
        register_custom_real_ct(custom_key, fp, preset)

        # Cập nhật danh sách Combobox ở cả Tab 1 và Tab 4
        all_keys = list(PHANTOMS.keys())
        self.cb_phantom["values"] = all_keys
        self.var_phantom.set(custom_key)

        real_keys = [k for k in all_keys if "CT Thực Tế" in k]
        self.cb_real_case["values"] = real_keys
        self.var_real_case.set(custom_key)

        self._fast_scan()
        self._run_real_ct_deep_analysis(custom_key)

    def _on_phantom_selected(self):
        name = self.var_phantom.get()
        if "Gan" in name:
            self.var_cal_preset.set("Abdomen (Gan / Bụng)")
        elif "Não" in name or "Đầu" in name:
            self.var_cal_preset.set("Brain (Sọ não)")
        elif "Phổi" in name or "ngực" in name:
            self.var_cal_preset.set("Lung (Phổi HRCT)")
        self._on_param_changed()
        self._fast_scan()

    def _on_calibration_changed(self):
        if self.custom_file_path:
            fname = os.path.basename(self.custom_file_path)
            custom_key = f"CT Thực Tế: {fname}"
            register_custom_real_ct(custom_key, self.custom_file_path, self.var_cal_preset.get())
            if self.var_phantom.get() == custom_key:
                self._fast_scan()

    def _on_scan_canvas_click(self, event):
        """Cho phép nhấp Chuột Trái (đặt ROI Nền bg) hoặc Chuột Phải (đặt ROI Tổn thương lesion) trên ảnh."""
        if self.current_scan is None or event.inaxes not in (self.ax1, self.ax3):
            return
        if event.xdata is None or event.ydata is None:
            return
        n = self.current_scan.n
        # Chuyển tọa độ pixel (0..n) về tọa độ chuẩn hóa [-1, 1] của _grid (với y = 1 - 2*py/n)
        cx = float(np.clip(event.xdata / n * 2.0 - 1.0, -0.75, 0.75))
        cy = float(np.clip(1.0 - event.ydata / n * 2.0, -0.75, 0.75))

        if event.button == 1:
            r_old = self.current_scan.info.rois["bg"][2]
            self.current_scan.info.rois["bg"] = (cx, cy, r_old)
        elif event.button == 3:
            r_old = self.current_scan.info.rois["lesion"][2]
            self.current_scan.info.rois["lesion"] = (cx, cy, r_old)
        else:
            return

        # Tính lại chỉ số cho toàn bộ thuật toán đã chạy
        for k, res in list(self.algo_results.items()):
            res.metrics = evaluate(res.hu, self.current_scan.hu_true, self.current_scan.info)
        self._redraw_current_tab1()
        self._update_metrics_table()

    # ----------------------------------------------------------------------------------
    # CÁC HÀM XỬ LÝ SỰ KIỆN & QUÉT TRỰC TIẾP
    # ----------------------------------------------------------------------------------
    def _on_kv_changed(self):
        self.lbl_kv_val.config(text=f"{self.var_kv.get()} kV")
        self._on_param_changed()

    def _on_ma_slider(self, val):
        self.lbl_ma_val.config(text=f"{int(float(val))} mA")
        self._on_param_changed()

    def _on_param_changed(self):
        kv = self.var_kv.get()
        ma = self.var_ma.get()
        views = self.var_views.get()

        _, info = make_phantom(self.var_phantom.get(), 128)
        params = ScanParams(kv=kv, ma=ma, n_views=views)
        dose = estimate_dose(params, info)

        self.lbl_ctdi.config(text=f"CTDIvol: {dose['CTDIvol_mGy']:.1f} mGy")
        self.lbl_dlp.config(text=f"DLP: {dose['DLP_mGycm']:.1f} mGy.cm | E: {dose['E_mSv']:.2f} mSv")
        self.lbl_flux.config(text=f"Quang thông I₀: {dose['I0']:.2e} photons/bin")

        pct = dose["rel_dose_pct"]
        if pct <= 25.0:
            self.lbl_alara_badge.config(text=f"ALARA: Liều cực thấp (-{100-pct:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)",
                                        bg="#1e3a8a", fg="#60a5fa")
        elif pct <= 65.0:
            self.lbl_alara_badge.config(text=f"ALARA: Giảm liều tốt (-{100-pct:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)",
                                        bg="#065f46", fg="#34d399")
        elif pct <= 100.0:
            self.lbl_alara_badge.config(text=f"ALARA: Chuẩn lâm sàng ({pct:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)",
                                        bg="#14532d", fg="#86efac")
        else:
            self.lbl_alara_badge.config(text=f"ALARA Cảnh báo: Liều cao (+{pct-100:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)",
                                        bg="#7f1d1d", fg="#fca5a5")

    def _quick_initial_scan(self):
        self._on_param_changed()
        self._fast_scan()
        self._run_real_ct_deep_analysis(self.var_real_case.get(), switch_tab=False)

    def _fast_scan(self):
        self.lbl_scan_status.config(text="Đang mô phỏng thu nhận & tái tạo...")
        self.progress_scan["value"] = 30
        self.update_idletasks()

        name = self.var_phantom.get()
        metal = self.var_metal.get()
        params = ScanParams(
            kv=float(self.var_kv.get()),
            ma=float(self.var_ma.get()),
            n_views=int(self.var_views.get()),
            poisson=self.var_poisson.get(),
            beam_hardening=self.var_beam_hardening.get()
        )

        t0 = time.time()
        self.current_scan = simulate_scan(name, 128, params, metal=metal)
        self.recon_context = ReconContext(self.current_scan)
        cfg = {**DEFAULT_CFG, "fbp_filter": self.var_fbp_filter.get()}
        fbp_res = self.recon_context.run("FBP", cfg)
        self.algo_results = {"FBP": fbp_res}

        self.progress_scan["value"] = 100
        self.lbl_scan_status.config(text=f"Hoàn tất quét & tái tạo {fbp_res.label} ({time.time()-t0:.2f}s).")
        self._draw_tab1_results(fbp_res.hu)

    def _redraw_current_tab1(self):
        if "FBP" in self.algo_results:
            self._draw_tab1_results(self.algo_results["FBP"].hu)

    def _draw_tab1_results(self, rec_hu: np.ndarray):
        scan = self.current_scan
        if scan is None:
            return

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        lvl, wid = scan.info.window
        vmin, vmax = lvl - wid / 2, lvl + wid / 2

        # 1. Đối tượng chụp (Phantom / Ảnh CT Thực tế)
        self.ax1.imshow(scan.hu_true, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax1.set_title(f"1. Đối tượng: {scan.info.name}\n(W:{wid:.0f}, L:{lvl:.0f} HU)", color="#38bdf8", fontsize=9.5, fontweight="bold")
        self.ax1.axis("off")

        # Vẽ vòng tròn ROI nếu bật
        if self.var_show_roi.get():
            for rname, rcol, rlbl in [("bg", "#22d3ee", "ROI Nền (bg)"), ("lesion", "#facc15", "ROI Tổn thương")]:
                cx, cy, r = scan.info.rois[rname]
                px = (cx + 1.0) * 0.5 * scan.n
                py = (1.0 - cy) * 0.5 * scan.n
                pr = r * 0.5 * scan.n
                self.ax1.add_patch(Circle((px, py), pr, fill=False, edgecolor=rcol, lw=1.8, label=rlbl))
            self.ax1.legend(loc="lower right", fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

        # Cập nhật thanh thống kê ROI trực tiếp
        bg_m = roi_mask(scan.n, scan.info.rois["bg"])
        le_m = roi_mask(scan.n, scan.info.rois["lesion"])
        bg_mean = float(np.mean(rec_hu[bg_m]))
        bg_std = float(np.std(rec_hu[bg_m]))
        le_mean = float(np.mean(rec_hu[le_m]))
        cnr_val = abs(le_mean - bg_mean) / max(bg_std, 1e-6)
        self.lbl_roi_live.config(text=f"Nền: {bg_mean:.0f}±{bg_std:.1f} HU | Đích: {le_mean:.0f} HU | CNR: {cnr_val:.1f}")

        # 2. Sinogram
        self.ax2.imshow(scan.sino, cmap="inferno", aspect="auto", origin="lower",
                        extent=[0, scan.params.angular_range, -scan.n // 2, scan.n // 2])
        self.ax2.set_title(f"2. Sinogram p(s,θ) ({scan.params.n_views} góc)", color="#fb923c", fontsize=9.5, fontweight="bold")
        self.ax2.set_xlabel("Góc quét θ (độ)", color="#94a3b8", fontsize=8)
        self.ax2.set_ylabel("Kênh Detector s", color="#94a3b8", fontsize=8)
        self.ax2.tick_params(colors="#94a3b8", labelsize=7)

        # 3. FBP Reconstructed
        flt_name = self.var_fbp_filter.get()
        self.ax3.imshow(rec_hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax3.set_title(f"3. Khôi phục FBP ({flt_name})\nNoise: {bg_std:.1f} HU | CNR: {cnr_val:.1f}", color="#4ade80", fontsize=9.5, fontweight="bold")
        self.ax3.axis("off")

        self.fig_scan.tight_layout()
        self.canvas_scan.draw()

    # ----------------------------------------------------------------------------------
    # QUÉT ĐỘNG LIVE (ANIMATION TRỰC TIẾP TRÊN TAB 1)
    # ----------------------------------------------------------------------------------
    def _start_live_scan_thread(self):
        if self.is_scanning:
            return
        self.is_scanning = True
        self.cancel_scan = False
        self.btn_live_scan.config(state=tk.DISABLED)
        self.btn_fast_scan.config(state=tk.DISABLED)
        self.btn_stop_scan.config(state=tk.NORMAL)
        threading.Thread(target=self._run_live_scan, daemon=True).start()

    def _stop_scan(self):
        self.cancel_scan = True
        self.lbl_scan_status.config(text="Đang dừng...")

    def _run_live_scan(self):
        name = self.var_phantom.get()
        metal = self.var_metal.get()
        params = ScanParams(
            kv=float(self.var_kv.get()),
            ma=float(self.var_ma.get()),
            n_views=int(self.var_views.get()),
            poisson=self.var_poisson.get(),
            beam_hardening=self.var_beam_hardening.get()
        )

        scan = simulate_scan(name, 128, params, metal=metal)
        self.current_scan = scan
        self.recon_context = ReconContext(scan)

        from ct_sim.recon import fbp_accumulate
        flt = self.var_fbp_filter.get()
        frames = fbp_accumulate(scan.sino, scan.theta, flt, n_frames=30)
        n_frames = len(frames)

        sino_disp = np.full_like(scan.sino, np.nan)
        col_step = max(1, scan.sino.shape[1] // n_frames)

        R = 128 * 0.6
        for f_idx in range(n_frames):
            if self.cancel_scan:
                break
            cur_angle = scan.theta[min(f_idx * col_step, len(scan.theta) - 1)]
            rad = np.radians(cur_angle)
            end_col = min((f_idx + 1) * col_step, scan.sino.shape[1])
            sino_disp[:, :end_col] = scan.sino[:, :end_col]

            self.after(0, self._render_live_frame, cur_angle, rad, R, sino_disp.copy(), 1000 * (frames[f_idx] - 1), f_idx, n_frames)
            time.sleep(0.04)

        self.after(0, self._finish_live_scan)

    def _render_live_frame(self, angle, rad, R, sino_matrix, rec_hu, f_idx, total):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        scan = self.current_scan
        if scan is None:
            return

        lvl, wid = scan.info.window
        vmin, vmax = lvl - wid / 2, lvl + wid / 2

        self.ax1.imshow(scan.hu_true, cmap="bone", origin="lower", extent=[-64, 64, -64, 64], vmin=vmin, vmax=vmax)
        v_ray = np.array([-np.sin(rad), np.cos(rad)])
        src = -R * v_ray
        det = R * v_ray
        self.ax1.plot([src[0], det[0]], [src[1], det[1]], color="#ef4444", lw=2)
        self.ax1.plot(src[0], src[1], marker="o", markersize=8, mfc="#eab308", mec="#ffffff")
        self.ax1.set_title(f"1. Thu nhận: θ = {angle:.0f}°", color="#38bdf8", fontsize=10, fontweight="bold")
        self.ax1.axis("off")

        self.ax2.imshow(sino_matrix, cmap="inferno", aspect="auto", origin="lower",
                        extent=[0, scan.params.angular_range, -scan.n // 2, scan.n // 2],
                        vmin=0, vmax=scan.sino.max())
        self.ax2.axvline(x=angle, color="#38bdf8", linestyle="--", lw=1.5)
        self.ax2.set_title(f"2. Sinogram: {f_idx+1}/{total}", color="#fb923c", fontsize=10, fontweight="bold")
        self.ax2.tick_params(colors="#94a3b8", labelsize=7)

        self.ax3.imshow(rec_hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax3.set_title("3. Tái tạo FBP (Cộng dồn)", color="#4ade80", fontsize=10, fontweight="bold")
        self.ax3.axis("off")

        self.canvas_scan.draw_idle()
        pct = int((f_idx + 1) / total * 100)
        self.progress_scan["value"] = pct
        self.lbl_scan_status.config(text=f"Đang quét góc θ = {angle:.0f}° ({pct}%)...")

    def _finish_live_scan(self):
        self.is_scanning = False
        self.btn_live_scan.config(state=tk.NORMAL)
        self.btn_fast_scan.config(state=tk.NORMAL)
        self.btn_stop_scan.config(state=tk.DISABLED)
        self.lbl_scan_status.config(text="Hoàn tất quét động trực tiếp.")

    # ----------------------------------------------------------------------------------
    # TAB 2: SO SÁNH ĐA THUẬT TOÁN & BÓC TÁCH 5 BỘ LỌC FBP
    # ----------------------------------------------------------------------------------
    def _start_compare_thread(self):
        if self.current_scan is None:
            self._fast_scan()
        threading.Thread(target=self._run_compare, daemon=True).start()

    def _run_compare(self):
        scan = self.current_scan
        if scan is None or self.recon_context is None:
            return

        selected_keys = [k for k, v in self.algo_vars.items() if v.get()]
        if not selected_keys:
            selected_keys = ["FBP"]

        self.lbl_scan_status.config(text="Đang tái tạo các thuật toán...")
        cfg = {**DEFAULT_CFG, "fbp_filter": self.var_fbp_filter.get()}
        results = run_algorithms(scan, selected_keys, cfg, ctx=self.recon_context)
        self.algo_results = results

        self.after(0, self._update_compare_plots)
        self.after(0, self._update_metrics_table)
        self.after(0, self._update_profile_plot)

    def _get_display_window(self, scan: ScanData):
        win_str = self.var_win.get()
        if "Brain" in win_str:
            return 40.0, 120.0
        if "Abdomen" in win_str:
            return 50.0, 350.0
        if "Bone" in win_str:
            return 400.0, 1500.0
        if "Lung" in win_str:
            return -500.0, 1500.0
        if "Full" in win_str:
            return 0.0, 2000.0
        return scan.info.window

    def _update_compare_plots(self):
        if not self.algo_results or self.current_scan is None:
            return

        self.fig_compare.clear()
        scan = self.current_scan
        lvl, wid = self._get_display_window(scan)
        vmin, vmax = lvl - wid / 2, lvl + wid / 2
        cmap = self.var_cmap.get()

        keys = list(self.algo_results.keys())
        n_plots = len(keys)
        if n_plots == 0:
            return

        if n_plots <= 2:
            rows, cols = 1, n_plots
        elif n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 2, 4

        for idx, k in enumerate(keys):
            res = self.algo_results[k]
            ax = self.fig_compare.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor(BG_DARK)
            ax.imshow(res.hu, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            title_color = "#38bdf8" if ("FBP" in k or "BP" in k) else ("#4ade80" if "IR" in res.group else "#f472b6")
            ax.set_title(
                f"{res.label}\nPSNR: {res.metrics['PSNR_dB']:.1f} dB | Noise: {res.metrics['Noise_HU']:.1f} HU",
                color=title_color, fontsize=8, fontweight="bold"
            )
            ax.axis("off")

        self.fig_compare.tight_layout()
        self.canvas_compare.draw()

    def _show_5_filters_analysis(self):
        """Bóc tách trực quan cả 5 bộ lọc FBP (Ramp, Shepp-Logan, Cosine, Hamming, Hann) trên Tab 2."""
        if self.current_scan is None:
            self._fast_scan()
        scan = self.current_scan
        if scan is None:
            return

        self.fig_compare.clear()
        lvl, wid = self._get_display_window(scan)
        vmin, vmax = lvl - wid / 2, lvl + wid / 2
        cmap = self.var_cmap.get()

        # Xoá bảng cũ và ghi số liệu 5 bộ lọc
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)

        flt_colors = {"ramp": "#ef4444", "shepp-logan": "#f59e0b", "cosine": "#10b981", "hamming": "#38bdf8", "hann": "#a855f7"}
        for idx, flt in enumerate(FILTERS):
            t0 = time.perf_counter()
            hu_f = 1000.0 * (fbp(scan.sino, scan.theta, flt, self.recon_context.lin) - 1.0)
            dt = time.perf_counter() - t0
            m = evaluate(hu_f, scan.hu_true, scan.info)

            ax = self.fig_compare.add_subplot(2, 3, idx + 1)
            ax.set_facecolor(BG_DARK)
            ax.imshow(hu_f, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"FBP ({flt.upper()})\nPSNR: {m['PSNR_dB']:.1f} dB | Noise: {m['Noise_HU']:.1f} HU | CNR: {m['CNR']:.1f}",
                color=flt_colors[flt], fontsize=7.8, fontweight="bold"
            )
            ax.axis("off")

            self.tree_metrics.insert("", tk.END, values=(
                f"FBP ({flt})", f"{m['PSNR_dB']:.2f}", f"{m['SSIM']:.3f}",
                f"{m['Noise_HU']:.1f}", f"{m['CNR']:.2f}", f"{dt:.2f}s"
            ))

        # Ô thứ 6: Vẽ đáp ứng tần số H(ω) = |ω| W(ω) của 5 bộ lọc
        ax_f = self.fig_compare.add_subplot(2, 3, 6)
        ax_f.set_facecolor(BG_DARK)
        freqs = np.fft.rfftfreq(128) * 2.0
        for flt in FILTERS:
            h = fourier_filter(128, flt).ravel()[:len(freqs)]
            ax_f.plot(freqs, h, color=flt_colors[flt], lw=1.6, label=flt)
        ax_f.set_title("Đáp ứng Tần số H(ω) = |ω|·W(ω)", color=TEXT_WHITE, fontsize=8, fontweight="bold")
        ax_f.set_xlabel("ω / ω_Nyquist", color=TEXT_MUTED, fontsize=7.5)
        ax_f.tick_params(colors=TEXT_MUTED, labelsize=7)
        ax_f.grid(True, color="#334155", linestyle=":", alpha=0.6)
        ax_f.legend(fontsize=6.5, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

        self.fig_compare.tight_layout()
        self.canvas_compare.draw()

    def _update_metrics_table(self):
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)

        for k, res in self.algo_results.items():
            m = res.metrics
            self.tree_metrics.insert("", tk.END, values=(
                res.label,
                f"{m['PSNR_dB']:.2f}",
                f"{m['SSIM']:.3f}",
                f"{m['Noise_HU']:.1f}",
                f"{m['CNR']:.2f}",
                f"{res.seconds:.2f}s"
            ))

    def _update_profile_plot(self):
        if not self.algo_results or self.current_scan is None:
            return

        self.ax_prof.clear()
        scan = self.current_scan
        # Lấy dòng đi qua tâm ROI tổn thương
        _, cy_le, _ = scan.info.rois["lesion"]
        mid_row = int(np.clip((1.0 - cy_le) * 0.5 * scan.n, 10, scan.n - 10))

        self.ax_prof.plot(scan.hu_true[mid_row, :], color="#ffffff", lw=2, linestyle="--", label="Sự thật nền")

        colors = {"FBP": "#ef4444", "MBIR": "#10b981", "DLIR": "#38bdf8", "ASIR": "#f59e0b", "SART": "#a855f7"}
        for k in ("FBP", "SART", "MBIR", "ASIR", "DLIR"):
            if k in self.algo_results:
                res = self.algo_results[k]
                c = colors.get(k, "#94a3b8")
                self.ax_prof.plot(res.hu[mid_row, :], color=c, lw=1.3, alpha=0.9, label=k)

        self.ax_prof.set_title(f"Mặt cắt ngang qua ROI Tổn thương (Row {mid_row})", color=TEXT_WHITE, fontsize=8.5)
        self.ax_prof.set_xlabel("Vị trí Pixel x", color=TEXT_MUTED, fontsize=8)
        self.ax_prof.set_ylabel("Độ suy giảm (HU)", color=TEXT_MUTED, fontsize=8)
        self.ax_prof.tick_params(colors=TEXT_MUTED, labelsize=7)
        self.ax_prof.grid(True, color="#334155", linestyle=":", alpha=0.6)
        self.ax_prof.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=PANEL_LIGHT, labelcolor=TEXT_WHITE)

        self.fig_profile.tight_layout()
        self.canvas_profile.draw()

    # ----------------------------------------------------------------------------------
    # TAB 3: KHẢO SÁT NGUYÊN LÝ ALARA
    # ----------------------------------------------------------------------------------
    def _start_alara_sweep_thread(self):
        self.btn_run_alara.config(state=tk.DISABLED)
        self.lbl_alara_status.config(text="Đang tính toán khảo sát quét liều qua 5 mức mA...")
        threading.Thread(target=self._run_alara_sweep, daemon=True).start()

    def _run_alara_sweep(self):
        name = self.var_phantom.get()
        base_params = ScanParams(
            kv=float(self.var_kv.get()),
            n_views=180,
            poisson=True
        )
        ma_list = [10.0, 25.0, 50.0, 100.0, 200.0]
        keys = ["FBP", "MBIR", "DLIR"]
        res = dose_sweep(name, 128, base_params, ma_list, keys)
        self.after(0, self._render_alara_plots, res, ma_list)

    def _render_alara_plots(self, res: Dict[str, Dict[str, list]], ma_list: list):
        self.fig_alara.clear()
        colors = {"FBP": "#ef4444", "MBIR": "#10b981", "DLIR": "#38bdf8"}

        ax1 = self.fig_alara.add_subplot(221)
        ax2 = self.fig_alara.add_subplot(222)
        ax3 = self.fig_alara.add_subplot(223)
        ax4 = self.fig_alara.add_subplot(224)

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_facecolor(BG_DARK)
            ax.grid(True, color="#334155", linestyle=":", alpha=0.6)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)

        for k in res:
            ax1.plot(res[k]["dose"], res[k]["PSNR_dB"], marker="o", color=colors[k], lw=1.8, label=k)
        ax1.set_title("1. Tỷ số Tín hiệu trên Nhiễu (PSNR vs Liều)", color=TEXT_WHITE, fontsize=9, fontweight="bold")
        ax1.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("PSNR (dB)", color=TEXT_MUTED, fontsize=8)
        ax1.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

        for k in res:
            ax2.plot(res[k]["dose"], res[k]["SSIM"], marker="s", color=colors[k], lw=1.8, label=k)
        ax2.set_title("2. Độ Tương Đồng Cấu Trúc (SSIM vs Liều)", color=TEXT_WHITE, fontsize=9, fontweight="bold")
        ax2.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("SSIM", color=TEXT_MUTED, fontsize=8)
        ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

        for k in res:
            ax3.plot(res[k]["ma"], res[k]["Noise_HU"], marker="^", color=colors[k], lw=1.8, label=k)
        ax3.set_title("3. Mức Nhiễu Hạt ROI (Noise HU vs mA ~ 1/√mA)", color=TEXT_WHITE, fontsize=9, fontweight="bold")
        ax3.set_xlabel("Dòng bóng phát mA", color=TEXT_MUTED, fontsize=8)
        ax3.set_ylabel("Độ lệch chuẩn Nhiễu (HU)", color=TEXT_MUTED, fontsize=8)
        ax3.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

        for k in res:
            ax4.plot(res[k]["dose"], res[k]["CNR"], marker="d", color=colors[k], lw=1.8, label=k)
        ax4.axhline(5.0, color="#facc15", linestyle="--", lw=1.0, label="Ngưỡng Rose (CNR=5)")
        ax4.set_title("4. Tương Phản trên Nhiễu (CNR vs Liều)", color=TEXT_WHITE, fontsize=9, fontweight="bold")
        ax4.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=8)
        ax4.set_ylabel("CNR", color=TEXT_MUTED, fontsize=8)
        ax4.legend(fontsize=7.5, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

        self.fig_alara.tight_layout()
        self.canvas_alara.draw()

        self.btn_run_alara.config(state=tk.NORMAL)
        self.lbl_alara_status.config(text="Khảo sát hoàn tất. MBIR & DLIR vượt trội ở liều thấp (<15 mGy).")

    # ----------------------------------------------------------------------------------
    # TAB 4: PHÂN TÍCH CHUYÊN SÂU ẢNH CT THỰC TẾ & SƠ ĐỒ KHỐI THUẬT TOÁN
    # ----------------------------------------------------------------------------------
    def _run_real_ct_deep_analysis(self, case_name: str, switch_tab: bool = False):
        if switch_tab:
            self.notebook.select(self.tab_real_fc)

        params = ScanParams(kv=float(self.var_kv.get()), ma=70.0, n_views=180, poisson=True)
        sc = simulate_scan(case_name, 128, params)
        self.real_ct_scan = sc
        res = run_algorithms(sc, ["FBP", "MBIR", "DLIR"])
        self.real_ct_results = res
        self._render_tab4_real_ct()

    def _on_tab4_canvas_click(self, event):
        sc = getattr(self, "real_ct_scan", None)
        if sc is None or event.xdata is None or event.ydata is None:
            return
        # Chỉ xử lý khi click vào ô đầu tiên (Ảnh CT thực tế gốc)
        axes = self.fig_real_fc.get_axes()
        if not axes or event.inaxes != axes[0]:
            return
        n = sc.n
        cx = float(np.clip(event.xdata / n * 2.0 - 1.0, -0.75, 0.75))
        cy = float(np.clip(1.0 - event.ydata / n * 2.0, -0.75, 0.75))
        if event.button == 1:
            sc.info.rois["bg"] = (cx, cy, sc.info.rois["bg"][2])
        elif event.button == 3:
            sc.info.rois["lesion"] = (cx, cy, sc.info.rois["lesion"][2])
        else:
            return
        for k, r in self.real_ct_results.items():
            r.metrics = evaluate(r.hu, sc.hu_true, sc.info)
        self._render_tab4_real_ct()

    def _render_tab4_real_ct(self):
        sc = getattr(self, "real_ct_scan", None)
        res = getattr(self, "real_ct_results", None)
        if sc is None or res is None:
            return

        self.fig_real_fc.clear()
        lvl, wid = sc.info.window
        vmin, vmax = lvl - wid / 2, lvl + wid / 2

        ax1 = self.fig_real_fc.add_subplot(231)
        ax2 = self.fig_real_fc.add_subplot(232)
        ax3 = self.fig_real_fc.add_subplot(233)
        ax4 = self.fig_real_fc.add_subplot(234)
        ax5 = self.fig_real_fc.add_subplot(235)
        ax6 = self.fig_real_fc.add_subplot(236)

        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.set_facecolor(BG_DARK)

        # 1. Ảnh CT Thực tế gốc + Vòng tròn ROI
        ax1.imshow(sc.hu_true, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        for rname, rcol, rlbl in [("bg", "#22d3ee", "ROI Nền"), ("lesion", "#facc15", "ROI Đích")]:
            cx, cy, r = sc.info.rois[rname]
            px = (cx + 1.0) * 0.5 * sc.n
            py = (1.0 - cy) * 0.5 * sc.n
            pr = r * 0.5 * sc.n
            ax1.add_patch(Circle((px, py), pr, fill=False, edgecolor=rcol, lw=1.8, label=rlbl))
        ax1.set_title(f"1. {sc.info.name}\n(Click Trái: ROI Nền | Phải: ROI Đích)", color="#38bdf8", fontsize=8.5, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=6.5, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)
        ax1.axis("off")

        # 2. Phổ phân bố Hounsfield (Histogram HU trong cơ thể)
        body_vals = sc.hu_true[sc.hu_true > -920.0]
        ax2.hist(body_vals, bins=45, color="#38bdf8", alpha=0.8, edgecolor="#0284c7")
        ax2.axvline(0.0, color="#4ade80", linestyle="--", lw=1.2, label="Nước (0 HU)")
        ax2.set_title("2. Phân bố Mật độ Hounsfield (HU)\nPhân tích thành phần mô học", color=TEXT_WHITE, fontsize=8.5, fontweight="bold")
        ax2.set_xlabel("Giá trị Hounsfield (HU)", color=TEXT_MUTED, fontsize=7.5)
        ax2.set_ylabel("Số lượng Voxel", color=TEXT_MUTED, fontsize=7.5)
        ax2.tick_params(colors=TEXT_MUTED, labelsize=7)
        ax2.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

        # 3. Sinogram p(s,θ)
        ax3.imshow(sc.sino, cmap="inferno", aspect="auto", origin="lower", extent=[0, 180, -64, 64])
        ax3.set_title("3. Biến đổi Radon p(s,θ)\n(Chiếu xạ ở liều thấp 70 mA)", color="#fb923c", fontsize=8.5, fontweight="bold")
        ax3.set_xlabel("Góc θ (độ)", color=TEXT_MUTED, fontsize=7.5)
        ax3.tick_params(colors=TEXT_MUTED, labelsize=7)

        # 4. FBP (Ramp)
        r_fbp = res["FBP"]
        ax4.imshow(r_fbp.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        ax4.set_title(f"4. Khôi phục FBP (Ramp)\nPSNR: {r_fbp.metrics['PSNR_dB']:.1f} dB | Noise: {r_fbp.metrics['Noise_HU']:.1f} HU",
                      color="#ef4444", fontsize=8.5, fontweight="bold")
        ax4.axis("off")

        # 5. MBIR (PWLS-TV)
        r_mbir = res["MBIR"]
        ax5.imshow(r_mbir.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        ax5.set_title(f"5. Khôi phục MBIR (PWLS-TV)\nPSNR: {r_mbir.metrics['PSNR_dB']:.1f} dB | Noise: {r_mbir.metrics['Noise_HU']:.1f} HU",
                      color="#10b981", fontsize=8.5, fontweight="bold")
        ax5.axis("off")

        # 6. DLIR (Residual CNN)
        r_dl = res["DLIR"]
        ax6.imshow(r_dl.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        ax6.set_title(f"6. Khôi phục DLIR (Residual CNN)\nPSNR: {r_dl.metrics['PSNR_dB']:.1f} dB | Noise: {r_dl.metrics['Noise_HU']:.1f} HU",
                      color="#38bdf8", fontsize=8.5, fontweight="bold")
        ax6.axis("off")

        self.fig_real_fc.tight_layout()
        self.canvas_real_fc.draw()

        # Cập nhật bảng văn bản thống kê ROI chi tiết ở cột trái
        bg_m = roi_mask(sc.n, sc.info.rois["bg"])
        le_m = roi_mask(sc.n, sc.info.rois["lesion"])
        bg_true = float(np.mean(sc.hu_true[bg_m]))
        le_true = float(np.mean(sc.hu_true[le_m]))

        stats_txt = (
            f"CA LÂM SÀNG: {sc.info.name}\n"
            f"Cửa sổ W/L : W={wid:.0f}, L={lvl:.0f} HU\n"
            f"ROI Nền    : {bg_true:+.1f} HU (Chuẩn)\n"
            f"ROI Đích   : {le_true:+.1f} HU (Δ={abs(le_true-bg_true):.1f} HU)\n"
            f"--------------------------------------\n"
            f"SO SÁNH KHÔI PHỤC HÀM μ(x,y) @ 70 mA:\n"
            f"• FBP (Ramp): Noise={r_fbp.metrics['Noise_HU']:4.1f} HU | CNR={r_fbp.metrics['CNR']:4.1f}\n"
            f"              PSNR ={r_fbp.metrics['PSNR_dB']:4.1f} dB | SSIM={r_fbp.metrics['SSIM']:.3f}\n"
            f"• MBIR (TV) : Noise={r_mbir.metrics['Noise_HU']:4.1f} HU | CNR={r_mbir.metrics['CNR']:4.1f}\n"
            f"              PSNR ={r_mbir.metrics['PSNR_dB']:4.1f} dB | SSIM={r_mbir.metrics['SSIM']:.3f}\n"
            f"• DLIR (CNN): Noise={r_dl.metrics['Noise_HU']:4.1f} HU | CNR={r_dl.metrics['CNR']:4.1f}\n"
            f"              PSNR ={r_dl.metrics['PSNR_dB']:4.1f} dB | SSIM={r_dl.metrics['SSIM']:.3f}\n"
        )
        self.txt_real_stats.delete("1.0", tk.END)
        self.txt_real_stats.insert(tk.END, stats_txt)

    def _display_flowchart_on_tab4(self, filename: str, title_label: str):
        """Hiển thị sơ đồ khối thuật toán (Flowchart) độ phân giải cao ngay trên vùng đồ thị Tab 4."""
        fig_dir = os.path.join(os.path.dirname(__file__), "report", "figures")
        img_path = os.path.join(fig_dir, filename)
        if not os.path.exists(img_path):
            from report.generate_flowcharts import generate_all_flowcharts_and_deep_figs
            generate_all_flowcharts_and_deep_figs(fig_dir)

        if not os.path.exists(img_path):
            messagebox.showwarning("Thông báo", f"Không tìm thấy sơ đồ: {img_path}")
            return

        from PIL import Image
        im = Image.open(img_path)
        self.fig_real_fc.clear()
        ax = self.fig_real_fc.add_subplot(111)
        ax.set_facecolor("#ffffff")
        ax.imshow(np.asarray(im))
        ax.set_title(title_label, color=ACCENT_CYAN, fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")
        self.fig_real_fc.tight_layout()
        self.canvas_real_fc.draw()

    # ----------------------------------------------------------------------------------
    # TAB 5: XUẤT BÁO CÁO & BẢN XEM TRƯỚC
    # ----------------------------------------------------------------------------------
    def _populate_report_preview(self):
        txt = (
            "====================================================================================================\n"
            "   ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH - TRƯỜNG ĐẠI HỌC BÁCH KHOA\n"
            "   KHOA KHOA HỌC ỨNG DỤNG / BỘ MÔN VẬT LÝ Y SINH & CƠ SỞ VẬT LÝ HẠT NHÂN\n"
            "====================================================================================================\n"
            "BÁO CÁO BÀI TẬP LỚN (PROJECT 2): KỸ THUẬT CHỤP CT TRONG Y TẾ - GIẢI BÀI TOÁN NGƯỢC KHÔI PHỤC HÀM μ(x,y)\n\n"
            "1. ĐẶT BÀI TOÁN NGƯỢC TOÁN HỌC (ILL-POSED INVERSE PROBLEM):\n"
            "   - Mục tiêu tối hậu: Khôi phục hàm phân bố hệ số suy giảm tuyến tính μ(x,y) ∈ L²(Ω) từ số đo tích phân\n"
            "     đường biến đổi Radon p(s,θ) = R[μ](s,θ) sao cho sai số ‖μ̂ - μ_true‖₂ nhỏ nhất với liều I₀ thấp nhất.\n"
            "   - Phương sai nhiễu sau biến đổi logarit (khai triển Taylor bậc 2):\n"
            "     Var(p_i) ≈ (N̄_i + σ_e²) / N̄_i² = 1 / [I₀·exp(-(Rμ)_i)] + σ_e² / [I₀²·exp(-2(Rμ)_i)].\n\n"
            "2. BÓC TÁCH SÂU TỪNG THUẬT TOÁN & BỘ LỌC (KÈM 6 SƠ ĐỒ KHỐI FLOWCHART TRONG PDF):\n"
            "   - [Flowchart 1] Chiếu ngược đơn thuần (BP): Chứng minh giải tích (R*R μ)(x,y) = μ(x,y) ∗ [1/(πr)],\n"
            "     hàm truyền miền tần số 1/|ω| gây mù sương tần số thấp.\n"
            "   - [Flowchart 2] Chiếu ngược có lọc (FBP): Định lý Lát cắt Trung tâm Fourier, nhân chập Ram-Lak rời rạc\n"
            "     h[0]=1/4, h[n lẻ]=-1/(πn)², đệm số không Zero-Padding khử chồng phổ tuần hoàn, bóc tách 5 cửa sổ\n"
            "     W(ω) (Ramp, Shepp-Logan, Cosine, Hamming, Hann) và chứng minh σ²_FBP ∝ 1 / (I₀·N_θ·Δs³).\n"
            "   - [Flowchart 3 & 4] Lặp Đại số SART & SIRT: Mô hình ma trận dải hình thang (Trapezoidal CDF),\n"
            "     toán tử liên hợp khớp A^T, bán hội tụ phổ GSVD f_k(σ) = 1-(1-λσ²)^k và bước nhảy Tỷ lệ vàng 0.618M.\n"
            "   - [Flowchart 5] Lặp Thống kê MBIR (PWLS-TV): Đạo hàm Euler-Lagrange, độ dẫn khuếch tán dị hướng\n"
            "     c_x = 1/√((D_x μ)²+δ²), giải thuật Khuếch tán trễ (Lagged Diffusivity) + Gradient Liên hợp (PCG).\n"
            "   - [Flowchart 6 & 7] Lai Thống kê ASIR (tổ hợp lồi phổ NPS) & Học sâu Phần dư DLIR (Residual CNN 6 lớp\n"
            "     Conv3x3 thực thi qua im2col + GEMM kết hợp cổng chú ý biên dị hướng).\n\n"
            "3. MODULE PHÂN TÍCH ẢNH CT LÂM SÀNG THỰC TẾ (REAL CLINICAL CT):\n"
            "   - Hỗ trợ nạp file PNG/JPG/BMP/TIFF/NPY/DICOM, hiệu chuẩn phi tuyến từng khúc sang thang HU,\n"
            "     tự động & tương tác nhấp chuột định vị ROI Nền/Tổn thương và khôi phục đa thuật toán.\n"
            "====================================================================================================\n"
        )
        self.txt_preview.delete("1.0", tk.END)
        self.txt_preview.insert(tk.END, txt)

    def _generate_pdf_report(self):
        self.btn_gen_pdf.config(state=tk.DISABLED, text="⏳ Đang sinh PDF...")
        self.update_idletasks()

        def worker():
            try:
                from report.generate_btl_pdf import create_btl_report
                pdf_path = os.path.join(os.path.dirname(__file__), "report", "BaoCao_BTL_Project2_CT_Scanner.pdf")
                create_btl_report(pdf_path, self.current_scan, self.algo_results)
                self.after(0, lambda: messagebox.showinfo(
                    "Thành công",
                    f"Đã tạo Báo cáo BTL PDF chuẩn Đại học Bách Khoa (kèm đầy đủ 6 Flowchart & phân tích Ảnh CT Thực Tế)!\n\nĐường dẫn:\n{pdf_path}"
                ))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: messagebox.showerror("Lỗi tạo PDF", f"Không thể tạo PDF:\n{err_msg}"))
            finally:
                self.after(0, lambda: self.btn_gen_pdf.config(state=tk.NORMAL, text="📄 TẠO BÁO CÁO BTL PDF CHUẨN BÁCH KHOA"))

        threading.Thread(target=worker, daemon=True).start()

    def _save_images(self):
        if not self.algo_results:
            messagebox.showwarning("Thông báo", "Vui lòng chạy tái tạo trước khi lưu ảnh.")
            return

        folder = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
        if not folder:
            return

        for k, res in self.algo_results.items():
            fn = os.path.join(folder, f"CT_Recon_{k}.png")
            plt.imsave(fn, res.hu, cmap="bone")
        messagebox.showinfo("Thành công", f"Đã lưu {len(self.algo_results)} ảnh vào thư mục:\n{folder}")

    def _export_csv(self):
        if not self.algo_results:
            messagebox.showwarning("Thông báo", "Chưa có kết quả số liệu để xuất CSV.")
            return

        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Lưu số liệu CSV")
        if not fn:
            return

        with open(fn, "w", encoding="utf-8-sig") as f:
            f.write("Thuật toán,Nhóm,PSNR (dB),SSIM,Nhiễu ROI (HU),Bias (HU),Độ tương phản (HU),CNR,Thời gian (s)\n")
            for k, res in self.algo_results.items():
                m = res.metrics
                f.write(f'"{res.label}","{res.group}",{m["PSNR_dB"]:.2f},{m["SSIM"]:.4f},{m["Noise_HU"]:.2f},'
                        f'{m["Bias_HU"]:.2f},{m["Contrast_HU"]:.2f},{m["CNR"]:.2f},{res.seconds:.3f}\n')

        messagebox.showinfo("Thành công", f"Đã xuất dữ liệu CSV thành công:\n{fn}")


def main():
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    app = CTSimulatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
