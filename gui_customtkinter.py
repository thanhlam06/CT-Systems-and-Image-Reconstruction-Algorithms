"""
gui.py - Trạm Làm Việc Y Tế Chuyên Nghiệp (Medical Pro Workstation)
Hệ Thống Mô Phỏng, Giải Bài Toán Ngược & Phân Tích Cắt Lớp Vi Tính (CT Simulator Pro).

Được xây dựng hoàn toàn trên nền tảng CustomTkinter hiện đại (Dark-Mode Card-Based UI),
kết hợp Matplotlib Dark Palette chuẩn các dòng máy chẩn đoán hình ảnh cao cấp (GE/Siemens).

Bao gồm 5 Tab tính năng cốt lõi:
  - Tab 1: ⚡ Trạm Quét Trực Tiếp & Nạp Ảnh CT (Flagship Workstation: Tỷ lệ 1:3, Pill Badges, 3 Subplots)
  - Tab 2: 📊 So Sánh 7 Thuật Toán Tái Tạo & Bóc Tách 5 Bộ Lọc FBP (Tần số / Không gian)
  - Tab 3: 📉 Khảo Sát Tối Ưu Hóa Liều Bức Xạ ALARA (Dose Sweep: mA = 10 -> 200)
  - Tab 4: 🔬 Phân Tích Chuyên Sâu CT Lâm Sàng Thực Tế & Trình Chiếu 6 Sơ Đồ Khối Toán Học (Flowcharts)
  - Tab 5: 📄 Xuất Báo Cáo Kỹ Thuật PDF Bách Khoa (21 Trang A4) & Dữ Liệu Thực Nghiệm
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Cấu hình Matplotlib sử dụng backend TkAgg tương thích CustomTkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# Thư viện giao diện người dùng hiện đại CustomTkinter
import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog

# ======================================================================================
# KẾT NỐI CÁC MODULE CỐT LÕI TỪ BACKEND PACKAGE ct_sim/
# ======================================================================================
# 1. Quản lý mô hình giải phẫu & ảnh CT lâm sàng thực tế
from ct_sim.phantoms import PHANTOMS, make_phantom, roi_mask, circle_mask, register_custom_real_ct
from ct_sim.real_ct import CALIBRATION_PRESETS, load_real_ct_image, ensure_builtin_real_ct_samples

# 2. Vật lý phát tia X, liều bức xạ ALARA & định luật Beer-Lambert
from ct_sim.physics import ScanParams, ScanData, simulate_scan, estimate_dose

# 3. Thuật toán tái tạo giải tích & bộ lọc FBP miền tần số
from ct_sim.recon import FILTERS, fourier_filter, fbp, fbp_accumulate

# 4. Trình điều phối thuật toán (BP, FBP, SART, SIRT, MBIR, ASIR, DLIR) & Quét liều
from ct_sim.pipeline import ALGOS, DEFAULT_CFG, ReconContext, run_algorithms, dose_sweep, label_for, AlgoResult

# 5. Hệ thống chỉ số đánh giá định lượng (RMSE, PSNR, SSIM, Noise ROI, CNR)
from ct_sim.metrics import evaluate

# ======================================================================================
# THIẾT LẬP BẢNG MÀU Y TẾ CAO CẤP (MEDICAL DARK SLATE PALETTE)
# ======================================================================================
APP_BG = "#090d16"          # Deep Navy Black (Nền toàn ứng dụng)
CARD_BG = "#111827"         # Dark Slate Card Surface (Bề mặt thẻ/khung chính)
CARD_SUB_BG = "#1e293b"     # Slate 800 Surface (Thẻ con / Bảng thông số phụ)
BORDER_COLOR = "#334155"    # Viền phân cách tế nhị
TEXT_MAIN = "#f8fafc"       # Chữ trắng sáng (Slate 50)
TEXT_MUTED = "#94a3b8"      # Chữ xám mờ (Slate 400)
ACCENT_CYAN = "#06b6d4"     # Neon Medical Cyan (Màu nhấn nút / Thanh trượt)
ACCENT_CYAN_HOVER = "#22d3ee"
ACCENT_EMERALD = "#10b981"  # Xanh an toàn / Tái tạo thành công (Safe ALARA)
ACCENT_AMBER = "#f59e0b"    # Vàng cảnh báo liều / Khối u ROI
ACCENT_ROSE = "#f43f5e"     # Đỏ tia X / Cảnh báo liều nguy hiểm
FONT_MAIN = "Segoe UI"

# Thiết lập chế độ Dark Mode mặc định cho CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ======================================================================================
# COMPONENT GIAO DIỆN PHỤ TRỢ: PILL BADGE (HUY HIỆU VIÊN THUỐC ĐO CHỈ SỐ)
# ======================================================================================
class MetricPillBadge(ctk.CTkFrame):
    """Component hiển thị chỉ số đo lường dạng viên thuốc bo tròn (Rounded Pill Badge)."""
    def __init__(self, parent, prefix: str, value: str,
                 accent_color: str = "#38bdf8", bg_color: str = CARD_SUB_BG,
                 border_color: str = BORDER_COLOR):
        super().__init__(parent, corner_radius=16, fg_color=bg_color,
                         border_width=1, border_color=border_color)
        self.prefix = prefix
        self.lbl = ctk.CTkLabel(
            self,
            text=f"{prefix}: {value}",
            font=(FONT_MAIN, 10, "bold"),
            text_color=accent_color
        )
        self.lbl.pack(padx=12, pady=4)

    def update_badge(self, value: str, accent_color: Optional[str] = None,
                     bg_color: Optional[str] = None, border_color: Optional[str] = None):
        """Cập nhật giá trị hiển thị và màu sắc trực tiếp theo thời gian thực."""
        self.lbl.configure(text=f"{self.prefix}: {value}")
        if accent_color:
            self.lbl.configure(text_color=accent_color)
        if bg_color:
            self.configure(fg_color=bg_color)
        if border_color:
            self.configure(border_color=border_color)


# ======================================================================================
# LỚP ỨNG DỤNG CHÍNH: CT SIMULATOR WORKSTATION (CUSTOMTKINTER)
# ======================================================================================
class CTSimulatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Cấu hình Cửa sổ & Hệ tọa độ Workstation
        self.title("CT SIMULATOR PRO — MEDICAL PRO WORKSTATION (BME LAB)")
        self.geometry("1440x850")
        self.minsize(1200, 750)
        self.configure(fg_color=APP_BG)

        # 2. Khởi tạo kho mẫu ảnh CT Thực tế trong assets/real_ct/
        ensure_builtin_real_ct_samples()

        # 3. Trạng thái dữ liệu nội bộ
        self.current_scan: Optional[ScanData] = None
        self.recon_context: Optional[ReconContext] = None
        self.algo_results: Dict[str, AlgoResult] = {}
        self.custom_file_path: Optional[str] = None
        self.is_scanning = False
        self.cancel_scan = False

        # 4. Cấu hình giao diện style cho các widget ttk con (Treeview...)
        self._init_ttk_styles()

        # 5. Xây dựng Thanh Tiêu Đề Thương Hiệu (Header Bar)
        self._create_header_bar()

        # 6. Tạo Tabview Hiện Đại 5 Phân Hệ (CTkTabview)
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=APP_BG,
            segmented_button_fg_color=CARD_SUB_BG,
            segmented_button_selected_color=ACCENT_CYAN,
            segmented_button_selected_hover_color=ACCENT_CYAN_HOVER,
            segmented_button_unselected_hover_color=BORDER_COLOR,
            text_color="#000000",
            corner_radius=12,
        )
        self.tabview.pack(fill="both", expand=True, padx=14, pady=(2, 8))

        # Khởi tạo 5 Tab chính
        self.tab_scan = self.tabview.add("⚡ 1. Quét & Tái Tạo (Workstation)")
        self.tab_compare = self.tabview.add("📊 2. So Sánh Thuật Toán & 5 Bộ Lọc FBP")
        self.tab_alara = self.tabview.add("📉 3. Khảo Sát Liều Bức Xạ ALARA")
        self.tab_real_fc = self.tabview.add("🔬 4. Phân Tích CT Thực Tế & Flowcharts")
        self.tab_export = self.tabview.add("📄 5. Báo Cáo BTL & Xuất Dữ Liệu")

        # 7. Xây dựng nội dung từng Tab
        self._build_tab_scan()
        self._build_tab_compare()
        self._build_tab_alara()
        self._build_tab_real_fc()
        self._build_tab_export()

        # 8. Kích hoạt mô phỏng lần đầu sau khi GUI nạp xong
        self.after(350, self._quick_initial_scan)

    # ----------------------------------------------------------------------------------
    # KHỞI TẠO STYLE & THANH TIÊU ĐỀ (HEADER)
    # ----------------------------------------------------------------------------------
    def _init_ttk_styles(self):
        """Định kiểu giao diện Dark Theme cho các widget bảng biểu ttk."""
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background=CARD_BG,
            foreground=TEXT_MAIN,
            fieldbackground=CARD_BG,
            rowheight=26,
            font=(FONT_MAIN, 9)
        )
        style.configure(
            "Treeview.Heading",
            background=CARD_SUB_BG,
            foreground="#38bdf8",
            font=(FONT_MAIN, 9, "bold"),
            relief="flat"
        )
        style.map("Treeview", background=[("selected", ACCENT_CYAN)], foreground=[("selected", "#000000")])

    def _create_header_bar(self):
        """Tạo thanh điều hướng & thương hiệu chuyên nghiệp ở đỉnh màn hình."""
        header = ctk.CTkFrame(self, fg_color=APP_BG, height=54, corner_radius=0)
        header.pack(fill="x", padx=16, pady=(8, 2))

        # Hiển thị Logo Đại học Bách Khoa
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "bk_logo.png")
        if os.path.exists(logo_path):
            try:
                pil_logo = Image.open(logo_path).resize((38, 38), Image.Resampling.LANCZOS)
                self.ctk_logo = ctk.CTkImage(light_image=pil_logo, dark_image=pil_logo, size=(38, 38))
                lbl_logo = ctk.CTkLabel(header, image=self.ctk_logo, text="")
                lbl_logo.pack(side="left", padx=(0, 10))
            except Exception:
                pass

        # Tiêu đề hệ thống & đơn vị đào tạo
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", fill="y")

        lbl_title = ctk.CTkLabel(
            title_box,
            text="CT SIMULATOR PRO — MEDICAL PRO WORKSTATION",
            font=(FONT_MAIN, 14, "bold"),
            text_color=TEXT_MAIN
        )
        lbl_title.pack(anchor="w")

        lbl_sub = ctk.CTkLabel(
            title_box,
            text="ĐHQG TP.HCM • Trường Đại học Bách khoa | Hệ thống Giải Bài toán Ngược & Khôi phục Hàm Suy giảm μ(x,y)",
            font=(FONT_MAIN, 10),
            text_color="#38bdf8"
        )
        lbl_sub.pack(anchor="w")

        # Huy hiệu trạng thái liều ALARA ở góc phải
        self.badge_header_alara = MetricPillBadge(
            header,
            prefix="TRẠNG THÁI ALARA",
            value="Chuẩn Lâm Sàng (120 mA | 27.0 mGy)",
            accent_color="#34d399",
            bg_color="#064e3b",
            border_color="#059669"
        )
        self.badge_header_alara.pack(side="right", padx=4)

    # ----------------------------------------------------------------------------------
    # TAB 1: ⚡ TRẠM QUÉT TRỰC TIẾP & NẠP ẢNH CT (WORKSTATION 1:3 RATIO)
    # ----------------------------------------------------------------------------------
    def _build_tab_scan(self):
        container = ctk.CTkFrame(self.tab_scan, fg_color=APP_BG)
        container.pack(fill="both", expand=True, padx=4, pady=4)

        # ==============================================================================
        # CỘT TRÁI (LEFT SIDEBAR - 25% WIDTH): BẢNG ĐIỀU KHIỂN DẠNG CUỘN
        # ==============================================================================
        self.sidebar_scan = ctk.CTkScrollableFrame(
            container,
            corner_radius=15,
            fg_color=CARD_BG,
            width=345,
            border_width=1,
            border_color=BORDER_COLOR
        )
        self.sidebar_scan.pack(side="left", fill="y", padx=(0, 8), pady=0)

        # --- Card 1: Chọn Phantom hoặc Nạp Ảnh CT Thực Tế ---
        card_p = ctk.CTkFrame(self.sidebar_scan, corner_radius=12, fg_color=CARD_SUB_BG)
        card_p.pack(fill="x", padx=6, pady=(6, 8))

        ctk.CTkLabel(card_p, text="1. PHANTOM & ẢNH CT THỰC TẾ", font=(FONT_MAIN, 11, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(8, 4))

        self.cb_phantom = ctk.CTkComboBox(
            card_p,
            values=list(PHANTOMS.keys()),
            command=lambda v: self._on_phantom_selected(),
            fg_color=CARD_BG,
            border_color=BORDER_COLOR,
            dropdown_fg_color=CARD_BG,
            height=30
        )
        self.cb_phantom.set("Đầu (HU)")
        self.cb_phantom.pack(fill="x", padx=10, pady=3)

        btn_browse_real = ctk.CTkButton(
            card_p,
            text="📂 Nạp Ảnh CT Thực Tế (PNG/DCM/NPY...)",
            font=(FONT_MAIN, 10, "bold"),
            fg_color="#eab308",
            hover_color="#facc15",
            text_color="#000000",
            height=30,
            command=self._browse_real_ct_file
        )
        btn_browse_real.pack(fill="x", padx=10, pady=(4, 6))

        # Cửa sổ hiệu chuẩn HU
        f_cal = ctk.CTkFrame(card_p, fg_color="transparent")
        f_cal.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f_cal, text="Hiệu chuẩn HU:", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left")
        self.cb_cal_preset = ctk.CTkComboBox(
            f_cal,
            values=list(CALIBRATION_PRESETS.keys()),
            width=175,
            height=26,
            command=lambda v: self._on_calibration_changed(),
            fg_color=CARD_BG,
            border_color=BORDER_COLOR,
            dropdown_fg_color=CARD_BG
        )
        self.cb_cal_preset.set("Auto (Tự động)")
        self.cb_cal_preset.pack(side="right")

        # Tùy chọn hiển thị vòng ROI & Cấy kim loại
        self.switch_show_roi = ctk.CTkSwitch(
            card_p,
            text="Hiện vòng ROI (L: Nền, R: U)",
            font=(FONT_MAIN, 9),
            text_color=TEXT_MAIN,
            progress_color=ACCENT_CYAN,
            command=self._redraw_current_tab1
        )
        self.switch_show_roi.select()
        self.switch_show_roi.pack(anchor="w", padx=10, pady=(6, 4))

        f_metal = ctk.CTkFrame(card_p, fg_color="transparent")
        f_metal.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(f_metal, text="Cấy kim loại:", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left")
        self.seg_metal = ctk.CTkSegmentedButton(
            f_metal,
            values=["Không", "1 Trụ", "2 Trụ"],
            selected_color=ACCENT_CYAN,
            selected_hover_color=ACCENT_CYAN_HOVER,
            command=lambda v: self._on_param_changed()
        )
        self.seg_metal.set("Không")
        self.seg_metal.pack(side="right")

        # --- Card 2: Thông Số Phát Tia X & Bộ Lọc FBP ---
        card_x = ctk.CTkFrame(self.sidebar_scan, corner_radius=12, fg_color=CARD_SUB_BG)
        card_x.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(card_x, text="2. THÔNG SỐ BÓNG PHÁT & BỘ LỌC", font=(FONT_MAIN, 11, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(8, 4))

        # Điện áp đỉnh (kVp)
        f_kv = ctk.CTkFrame(card_x, fg_color="transparent")
        f_kv.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f_kv, text="Điện áp đỉnh (kVp):", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left")
        self.lbl_kv_disp = ctk.CTkLabel(f_kv, text="120 kV", font=(FONT_MAIN, 10, "bold"), text_color=ACCENT_CYAN)
        self.lbl_kv_disp.pack(side="right")

        self.seg_kv = ctk.CTkSegmentedButton(
            card_x,
            values=["80 kV", "100 kV", "120 kV", "140 kV"],
            selected_color=ACCENT_CYAN,
            selected_hover_color=ACCENT_CYAN_HOVER,
            command=lambda v: self._on_kv_changed(v)
        )
        self.seg_kv.set("120 kV")
        self.seg_kv.pack(fill="x", padx=10, pady=(2, 6))

        # Cường độ dòng ống (mA)
        f_ma = ctk.CTkFrame(card_x, fg_color="transparent")
        f_ma.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f_ma, text="Dòng ống (mA):", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left")
        self.lbl_ma_disp = ctk.CTkLabel(f_ma, text="120 mA", font=(FONT_MAIN, 10, "bold"), text_color="#38bdf8")
        self.lbl_ma_disp.pack(side="right")

        self.slider_ma = ctk.CTkSlider(
            card_x,
            from_=10,
            to=300,
            number_of_steps=29,
            progress_color=ACCENT_CYAN,
            button_color=ACCENT_CYAN,
            button_hover_color=ACCENT_CYAN_HOVER,
            command=self._on_ma_slider
        )
        self.slider_ma.set(120)
        self.slider_ma.pack(fill="x", padx=10, pady=(2, 6))

        # Số góc chiếu (Views)
        f_vw = ctk.CTkFrame(card_x, fg_color="transparent")
        f_vw.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f_vw, text="Số góc chiếu (Views):", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left")
        self.cb_views = ctk.CTkComboBox(
            f_vw,
            values=["60", "90", "120", "180"],
            width=80,
            height=26,
            command=lambda v: self._on_param_changed(),
            fg_color=CARD_BG,
            border_color=BORDER_COLOR,
            dropdown_fg_color=CARD_BG
        )
        self.cb_views.set("180")
        self.cb_views.pack(side="right")

        # Cửa sổ bộ lọc FBP H(ω)
        f_flt = ctk.CTkFrame(card_x, fg_color="transparent")
        f_flt.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f_flt, text="Bộ lọc FBP H(ω):", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left")
        self.cb_fbp_filter = ctk.CTkComboBox(
            f_flt,
            values=list(FILTERS),
            width=115,
            height=26,
            command=lambda v: self._fast_scan(),
            fg_color=CARD_BG,
            border_color=BORDER_COLOR,
            dropdown_fg_color=CARD_BG
        )
        self.cb_fbp_filter.set("ramp")
        self.cb_fbp_filter.pack(side="right")

        # Checkbox Nhiễu Poisson & Cứng hóa chùm tia
        self.chk_poisson = ctk.CTkCheckBox(
            card_x,
            text="Nhiễu lượng tử Poisson",
            font=(FONT_MAIN, 9),
            text_color=TEXT_MAIN,
            fg_color=ACCENT_CYAN,
            command=self._on_param_changed
        )
        self.chk_poisson.select()
        self.chk_poisson.pack(anchor="w", padx=10, pady=(6, 2))

        self.chk_hardening = ctk.CTkCheckBox(
            card_x,
            text="Hiệu ứng Cứng hóa chùm tia",
            font=(FONT_MAIN, 9),
            text_color=TEXT_MAIN,
            fg_color=ACCENT_CYAN,
            command=self._on_param_changed
        )
        self.chk_hardening.pack(anchor="w", padx=10, pady=(2, 8))

        # --- Card 3: Nút Bấm Điều Khiển Quét & Tiến Trình ---
        card_ctrl = ctk.CTkFrame(self.sidebar_scan, corner_radius=12, fg_color=CARD_SUB_BG)
        card_ctrl.pack(fill="x", padx=6, pady=(4, 8))

        ctk.CTkLabel(card_ctrl, text="3. THAO TÁC QUÉT & TÁI TẠO", font=(FONT_MAIN, 11, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(8, 4))

        # Nút hành động chính (Primary Action Button)
        self.btn_fast_scan = ctk.CTkButton(
            card_ctrl,
            text="⚡ Quét Nhanh & Khôi Phục (FBP)",
            font=(FONT_MAIN, 11, "bold"),
            fg_color=ACCENT_CYAN,
            hover_color=ACCENT_CYAN_HOVER,
            text_color="#000000",
            height=34,
            command=self._fast_scan
        )
        self.btn_fast_scan.pack(fill="x", padx=10, pady=3)

        self.btn_live_scan = ctk.CTkButton(
            card_ctrl,
            text="▶ Quét Động Trực Tiếp (Live Scan)",
            font=(FONT_MAIN, 10, "bold"),
            fg_color=ACCENT_EMERALD,
            hover_color="#34d399",
            text_color="#ffffff",
            height=30,
            command=self._start_live_scan_thread
        )
        self.btn_live_scan.pack(fill="x", padx=10, pady=3)

        self.btn_stop_scan = ctk.CTkButton(
            card_ctrl,
            text="⏹ Dừng Quét",
            font=(FONT_MAIN, 10),
            fg_color="#475569",
            hover_color="#64748b",
            text_color="#ffffff",
            height=28,
            state="disabled",
            command=self._stop_scan
        )
        self.btn_stop_scan.pack(fill="x", padx=10, pady=3)

        self.progress_scan = ctk.CTkProgressBar(card_ctrl, progress_color=ACCENT_CYAN)
        self.progress_scan.set(0)
        self.progress_scan.pack(fill="x", padx=10, pady=(6, 2))

        self.lbl_scan_status = ctk.CTkLabel(
            card_ctrl,
            text="Sẵn sàng quét.",
            font=(FONT_MAIN, 9),
            text_color=TEXT_MUTED
        )
        self.lbl_scan_status.pack(anchor="w", padx=10, pady=(0, 8))

        # ==============================================================================
        # CỘT PHẢI (RIGHT VIEWPORT - 75% WIDTH): PILL BADGES & MATPLOTLIB WORKSTATION
        # ==============================================================================
        self.viewport_scan = ctk.CTkFrame(container, corner_radius=15, fg_color=APP_BG)
        self.viewport_scan.pack(side="left", fill="both", expand=True)

        # 1. Hàng Thẻ Pill Badges hiển thị chỉ số đo lường lâm sàng trên đỉnh
        self.pill_bar = ctk.CTkFrame(self.viewport_scan, corner_radius=14, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        self.pill_bar.pack(fill="x", padx=2, pady=(0, 6))

        self.pill_ctdi = MetricPillBadge(self.pill_bar, "CTDIvol", "27.0 mGy", "#38bdf8")
        self.pill_ctdi.pack(side="left", padx=4, pady=4)

        self.pill_dlp = MetricPillBadge(self.pill_bar, "DLP & E", "405.0 mGy·cm | 0.85 mSv", "#4ade80")
        self.pill_dlp.pack(side="left", padx=4, pady=4)

        self.pill_flux = MetricPillBadge(self.pill_bar, "Quang Thông I₀", "1.33e+06 ph/bin", "#94a3b8")
        self.pill_flux.pack(side="left", padx=4, pady=4)

        self.pill_roi = MetricPillBadge(self.pill_bar, "ROI Nền / Đích", "-- HU | -- HU", "#facc15")
        self.pill_roi.pack(side="left", padx=4, pady=4)

        self.pill_cnr = MetricPillBadge(self.pill_bar, "CNR / Noise", "-- | -- HU", "#f472b6")
        self.pill_cnr.pack(side="left", padx=4, pady=4)

        # 2. Thẻ Lớn Chứa Khung Vẽ 3 Subplots Matplotlib (Card-Based Viewport)
        self.card_canvas = ctk.CTkFrame(self.viewport_scan, corner_radius=15, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        self.card_canvas.pack(fill="both", expand=True, padx=2, pady=0)

        # Áp dụng Dark Medical Theme vào Matplotlib Figure
        self.fig_scan = Figure(figsize=(10, 5.0), dpi=100, facecolor=CARD_BG)
        self.ax1 = self.fig_scan.add_subplot(131, facecolor=CARD_BG)
        self.ax2 = self.fig_scan.add_subplot(132, facecolor=CARD_BG)
        self.ax3 = self.fig_scan.add_subplot(133, facecolor=CARD_BG)

        for ax in (self.ax1, self.ax2, self.ax3):
            self._style_axis_dark(ax)

        self.fig_scan.tight_layout()
        self.canvas_scan = FigureCanvasTkAgg(self.fig_scan, master=self.card_canvas)
        self.canvas_scan.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        # Bắt sự kiện Click chuột Trái/Phải để di chuyển tâm ROI
        self.canvas_scan.mpl_connect("button_press_event", self._on_scan_canvas_click)

    # ----------------------------------------------------------------------------------
    # TAB 2: 📊 SO SÁNH 7 THUẬT TOÁN TÁI TẠO & 5 BỘ LỌC FBP
    # ----------------------------------------------------------------------------------
    def _build_tab_compare(self):
        container = ctk.CTkFrame(self.tab_compare, fg_color=APP_BG)
        container.pack(fill="both", expand=True, padx=4, pady=4)

        # Thanh điều khiển phía trên
        top_bar = ctk.CTkFrame(container, corner_radius=12, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        top_bar.pack(fill="x", padx=2, pady=(0, 6))

        ctk.CTkLabel(top_bar, text="Thuật toán:", font=(FONT_MAIN, 10, "bold"), text_color=ACCENT_CYAN).pack(side="left", padx=(10, 6), pady=8)

        self.algo_vars: Dict[str, ctk.CTkCheckBox] = {}
        for key in ("BP", "FBP", "SART", "SIRT", "MBIR", "ASIR", "DLIR"):
            chk = ctk.CTkCheckBox(
                top_bar,
                text=key,
                font=(FONT_MAIN, 9),
                fg_color=ACCENT_CYAN,
                text_color=TEXT_MAIN,
                width=55
            )
            if key in ("BP", "FBP", "SART", "MBIR", "ASIR", "DLIR"):
                chk.select()
            chk.pack(side="left", padx=3)
            self.algo_vars[key] = chk

        btn_run_cmp = ctk.CTkButton(
            top_bar,
            text="▶ Tái Tạo & Đo Lường",
            font=(FONT_MAIN, 10, "bold"),
            fg_color=ACCENT_CYAN,
            hover_color=ACCENT_CYAN_HOVER,
            text_color="#000000",
            height=28,
            command=self._start_compare_thread
        )
        btn_run_cmp.pack(side="left", padx=8)

        btn_flt_study = ctk.CTkButton(
            top_bar,
            text="🔍 Bóc Tách 5 Bộ Lọc FBP",
            font=(FONT_MAIN, 10, "bold"),
            fg_color=ACCENT_AMBER,
            hover_color="#fbbf24",
            text_color="#000000",
            height=28,
            command=self._show_5_filters_analysis
        )
        btn_flt_study.pack(side="left", padx=4)

        # Cửa sổ & Colormap
        f_opt = ctk.CTkFrame(top_bar, fg_color="transparent")
        f_opt.pack(side="right", padx=10, pady=6)

        ctk.CTkLabel(f_opt, text="Cửa sổ:", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left", padx=3)
        self.cb_win = ctk.CTkComboBox(
            f_opt,
            values=["Auto (Theo Phantom)", "Brain (L:40, W:120)", "Abdomen (L:50, W:350)", "Bone (L:400, W:1500)", "Lung (L:-500, W:1500)", "Full (-1000, 1000)"],
            width=165,
            height=26,
            command=lambda v: self._update_compare_plots(),
            fg_color=CARD_SUB_BG,
            border_color=BORDER_COLOR
        )
        self.cb_win.set("Auto (Theo Phantom)")
        self.cb_win.pack(side="left", padx=2)

        ctk.CTkLabel(f_opt, text="Màu:", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(side="left", padx=(6, 3))
        self.cb_cmap = ctk.CTkComboBox(
            f_opt,
            values=["bone", "gray", "inferno", "magma", "viridis"],
            width=85,
            height=26,
            command=lambda v: self._update_compare_plots(),
            fg_color=CARD_SUB_BG,
            border_color=BORDER_COLOR
        )
        self.cb_cmap.set("bone")
        self.cb_cmap.pack(side="left", padx=2)

        # Phần thân chia 2: Lưới ảnh bên trái + Bảng số liệu & Mặt cắt 1D bên phải
        split_frame = ctk.CTkFrame(container, fg_color=APP_BG)
        split_frame.pack(fill="both", expand=True)

        # Khung chứa hình ảnh tái tạo
        img_card = ctk.CTkFrame(split_frame, corner_radius=12, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        img_card.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.fig_compare = Figure(figsize=(7.6, 6.0), dpi=100, facecolor=CARD_BG)
        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, master=img_card)
        self.canvas_compare.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        # Cột phụ bên phải: Bảng số liệu định lượng + Đồ thị mặt cắt 1D qua tổn thương
        side_card = ctk.CTkFrame(split_frame, corner_radius=12, fg_color=CARD_SUB_BG, width=440, border_width=1, border_color=BORDER_COLOR)
        side_card.pack(side="right", fill="both", padx=(4, 0))
        side_card.pack_propagate(False)

        ctk.CTkLabel(side_card, text="BẢNG CHỈ SỐ ĐỊNH LƯỢNG KHÔI PHỤC HÀM μ(x,y)", font=(FONT_MAIN, 10, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(8, 4))

        cols = ("algo", "psnr", "ssim", "noise", "cnr", "time")
        self.tree_metrics = ttk.Treeview(side_card, columns=cols, show="headings", height=8)
        self.tree_metrics.heading("algo", text="Thuật toán / Bộ lọc")
        self.tree_metrics.heading("psnr", text="PSNR (dB)")
        self.tree_metrics.heading("ssim", text="SSIM")
        self.tree_metrics.heading("noise", text="Nhiễu (HU)")
        self.tree_metrics.heading("cnr", text="CNR")
        self.tree_metrics.heading("time", text="Thời gian")

        self.tree_metrics.column("algo", width=120, anchor="w")
        self.tree_metrics.column("psnr", width=65, anchor="center")
        self.tree_metrics.column("ssim", width=55, anchor="center")
        self.tree_metrics.column("noise", width=68, anchor="center")
        self.tree_metrics.column("cnr", width=55, anchor="center")
        self.tree_metrics.column("time", width=62, anchor="center")
        self.tree_metrics.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(side_card, text="MẶT CẮT ĐƯỜNG NGANG 1D QUA TỔN THƯƠNG (HU PROFILE)", font=(FONT_MAIN, 10, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(8, 2))

        self.fig_profile = Figure(figsize=(4.3, 3.0), dpi=90, facecolor=CARD_SUB_BG)
        self.ax_prof = self.fig_profile.add_subplot(111, facecolor=CARD_BG)
        self._style_axis_dark(self.ax_prof)
        self.fig_profile.tight_layout()

        self.canvas_profile = FigureCanvasTkAgg(self.fig_profile, master=side_card)
        self.canvas_profile.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ----------------------------------------------------------------------------------
    # TAB 3: 📉 KHẢO SÁT LIỀU BỨC XẠ ALARA
    # ----------------------------------------------------------------------------------
    def _build_tab_alara(self):
        container = ctk.CTkFrame(self.tab_alara, fg_color=APP_BG)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        top_card = ctk.CTkFrame(container, corner_radius=12, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        top_card.pack(fill="x", padx=2, pady=(0, 6))

        ctk.CTkLabel(
            top_card,
            text="Khảo sát suy giảm chất lượng ảnh theo dòng bóng phát (mA) & liều hấp thụ ALARA (Kiểm chứng định luật σ ~ 1/√mA):",
            font=(FONT_MAIN, 10, "bold"),
            text_color=ACCENT_CYAN
        ).pack(side="left", padx=12, pady=10)

        self.btn_run_alara = ctk.CTkButton(
            top_card,
            text="⚡ Chạy Quét Đồ Thị Liều (mA = 10 -> 200)",
            font=(FONT_MAIN, 10, "bold"),
            fg_color=ACCENT_AMBER,
            hover_color="#fbbf24",
            text_color="#000000",
            height=30,
            command=self._start_alara_sweep_thread
        )
        self.btn_run_alara.pack(side="left", padx=12)

        self.lbl_alara_status = ctk.CTkLabel(
            top_card,
            text="Nhấn nút để chạy khảo sát trên đối tượng hiện tại.",
            font=(FONT_MAIN, 9),
            text_color=TEXT_MUTED
        )
        self.lbl_alara_status.pack(side="left", padx=6)

        # Khung đồ thị 4 Subplots ALARA
        plot_card = ctk.CTkFrame(container, corner_radius=12, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        plot_card.pack(fill="both", expand=True)

        self.fig_alara = Figure(figsize=(10, 5.8), dpi=100, facecolor=CARD_BG)
        self.canvas_alara = FigureCanvasTkAgg(self.fig_alara, master=plot_card)
        self.canvas_alara.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    # ----------------------------------------------------------------------------------
    # TAB 4: 🔬 PHÂN TÍCH CT THỰC TẾ & FLOWCHARTS TOÁN HỌC
    # ----------------------------------------------------------------------------------
    def _build_tab_real_fc(self):
        container = ctk.CTkFrame(self.tab_real_fc, fg_color=APP_BG)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        # Cột điều khiển bên trái
        left_box = ctk.CTkScrollableFrame(container, corner_radius=14, fg_color=CARD_BG, width=340, border_width=1, border_color=BORDER_COLOR)
        left_box.pack(side="left", fill="y", padx=(0, 8), pady=0)

        ctk.CTkLabel(left_box, text="1. KHO ẢNH CT LÂM SÀNG THỰC TẾ", font=(FONT_MAIN, 11, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(10, 4))

        real_presets = [k for k in PHANTOMS.keys() if "CT Thực Tế" in k]
        default_case = real_presets[0] if real_presets else "CT Thực Tế: Gan Bình Thường"

        self.cb_real_case = ctk.CTkComboBox(
            left_box,
            values=real_presets,
            command=lambda v: self._run_real_ct_deep_analysis(v),
            fg_color=CARD_SUB_BG,
            border_color=BORDER_COLOR,
            dropdown_fg_color=CARD_BG,
            height=30
        )
        self.cb_real_case.set(default_case)
        self.cb_real_case.pack(fill="x", padx=10, pady=2)

        btn_open_custom = ctk.CTkButton(
            left_box,
            text="📂 Mở File Ảnh CT Thực Tế Bất Kỳ...",
            font=(FONT_MAIN, 10, "bold"),
            fg_color="#eab308",
            hover_color="#facc15",
            text_color="#000000",
            height=30,
            command=self._browse_real_ct_file
        )
        btn_open_custom.pack(fill="x", padx=10, pady=(6, 3))

        btn_run_real = ctk.CTkButton(
            left_box,
            text="🔬 Phân Tích Chuyên Sâu Ca CT Này",
            font=(FONT_MAIN, 10, "bold"),
            fg_color=ACCENT_CYAN,
            hover_color=ACCENT_CYAN_HOVER,
            text_color="#000000",
            height=30,
            command=lambda: self._run_real_ct_deep_analysis(self.cb_real_case.get())
        )
        btn_run_real.pack(fill="x", padx=10, pady=3)

        # Khung thông tin mô học & định lượng ROI
        ctk.CTkLabel(left_box, text="2. THỐNG KÊ MÔ HỌC & ĐỊNH LƯỢNG ROI", font=(FONT_MAIN, 11, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(12, 4))
        self.txt_real_stats = ctk.CTkTextbox(
            left_box,
            height=145,
            corner_radius=10,
            fg_color=CARD_SUB_BG,
            text_color=TEXT_MAIN,
            font=("Consolas", 9)
        )
        self.txt_real_stats.pack(fill="x", padx=10, pady=2)

        # Trình chiếu 6 Sơ đồ khối thuật toán (Flowcharts)
        ctk.CTkLabel(left_box, text="3. SƠ ĐỒ KHỐI TOÁN HỌC (FLOWCHARTS)", font=(FONT_MAIN, 11, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=10, pady=(12, 4))
        ctk.CTkLabel(left_box, text="Nhấp chọn sơ đồ để phóng to trực tiếp:", font=(FONT_MAIN, 9), text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(0, 4))

        self.flowchart_options = [
            ("Flowchart 0: Khung Bài Toán Ngược", "fc0_inverse_problem_framework.png"),
            ("Flowchart 1: Chiếu Ngược Đơn Thuần (BP)", "fc1_bp_flowchart.png"),
            ("Flowchart 2: Chiếu Ngược Có Lọc (FBP)", "fc2_fbp_flowchart.png"),
            ("Flowchart 3 & 4: Lặp Đại Số SART & SIRT", "fc3_sart_sirt_flowchart.png"),
            ("Flowchart 5: Lặp Thống Kê MBIR (PWLS-TV)", "fc4_mbir_pwls_tv_flowchart.png"),
            ("Flowchart 6 & 7: Lai ASIR & Học Sâu DLIR", "fc5_asir_dlir_flowchart.png"),
            ("Đồ Thị Phân Tích 5 Bộ Lọc FBP", "fig5_filter_frequency_spatial.png"),
        ]
        for label_txt, fname in self.flowchart_options:
            btn = ctk.CTkButton(
                left_box,
                text="📐 " + label_txt,
                anchor="w",
                font=(FONT_MAIN, 9),
                fg_color=CARD_SUB_BG,
                hover_color=ACCENT_CYAN,
                text_color=TEXT_MAIN,
                height=26,
                command=lambda fn=fname, lt=label_txt: self._display_flowchart_on_tab4(fn, lt)
            )
            btn.pack(fill="x", padx=10, pady=2)

        # Khung hiển thị bên phải của Tab 4
        right_box = ctk.CTkFrame(container, corner_radius=14, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        right_box.pack(side="left", fill="both", expand=True)

        self.fig_real_fc = Figure(figsize=(10.2, 6.0), dpi=100, facecolor=CARD_BG)
        self.canvas_real_fc = FigureCanvasTkAgg(self.fig_real_fc, master=right_box)
        self.canvas_real_fc.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas_real_fc.mpl_connect("button_press_event", self._on_tab4_canvas_click)

    # ----------------------------------------------------------------------------------
    # TAB 5: 📄 BÁO CÁO BTL & XUẤT DỮ LIỆU
    # ----------------------------------------------------------------------------------
    def _build_tab_export(self):
        container = ctk.CTkFrame(self.tab_export, fg_color=APP_BG)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        info_panel = ctk.CTkFrame(container, corner_radius=14, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        info_panel.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            info_panel,
            text="XUẤT BÁO CÁO BÀI TẬP LỚN PDF CHUẨN ĐẠI HỌC BÁCH KHOA (KÈM 6 FLOWCHARTS & TOÁN HỌC CHUYÊN SÂU)",
            font=(FONT_MAIN, 11, "bold"),
            text_color=ACCENT_CYAN
        ).pack(anchor="w", padx=16, pady=(12, 4))

        desc_text = (
            "Hệ thống tự động kết xuất toàn bộ báo cáo BTL định dạng PDF chuẩn Bách Khoa (bám sát mẫu BTLXSTK_Khoa Cơ Khí_L01.pdf), "
            "bao gồm: Khung lý thuyết Bài toán ngược đặt không chỉnh (Ill-posed Inverse Problem), chứng minh hàm truyền điểm 1/(πr) của BP, "
            "bóc tách nhân Ram-Lak & định lý truyền phương sai nhiễu của 5 bộ lọc FBP, mô hình dải hình thang & bán hội tụ GSVD của SART/SIRT, "
            "đạo hàm Euler-Lagrange & giải thuật Lagged Diffusivity + PCG của MBIR, điều khiển phổ NPS của ASIR, mạng Residual CNN của DLIR, "
            "6 Flowchart thuật toán và kết quả phân tích Ảnh CT Lâm sàng Thực tế."
        )
        ctk.CTkLabel(
            info_panel,
            text=desc_text,
            font=(FONT_MAIN, 9),
            text_color=TEXT_MUTED,
            justify="left",
            wraplength=1250
        ).pack(anchor="w", padx=16, pady=(0, 10))

        btn_bar = ctk.CTkFrame(info_panel, fg_color="transparent")
        btn_bar.pack(fill="x", padx=16, pady=(0, 12))

        self.btn_gen_pdf = ctk.CTkButton(
            btn_bar,
            text="📄 TẠO BÁO CÁO BTL PDF CHUẨN BÁCH KHOA",
            font=(FONT_MAIN, 11, "bold"),
            fg_color=ACCENT_EMERALD,
            hover_color="#34d399",
            text_color="#ffffff",
            height=34,
            command=self._generate_pdf_report
        )
        self.btn_gen_pdf.pack(side="left", padx=(0, 10))

        self.btn_save_images = ctk.CTkButton(
            btn_bar,
            text="💾 Lưu Toàn Bộ Ảnh Tái Tạo (PNG)",
            font=(FONT_MAIN, 10),
            fg_color=CARD_SUB_BG,
            hover_color=BORDER_COLOR,
            text_color=TEXT_MAIN,
            height=34,
            command=self._save_images
        )
        self.btn_save_images.pack(side="left", padx=6)

        self.btn_export_csv = ctk.CTkButton(
            btn_bar,
            text="📊 Xuất Bảng Số Liệu (CSV)",
            font=(FONT_MAIN, 10),
            fg_color=CARD_SUB_BG,
            hover_color=BORDER_COLOR,
            text_color=TEXT_MAIN,
            height=34,
            command=self._export_csv
        )
        self.btn_export_csv.pack(side="left", padx=6)

        # Hộp văn bản tóm tắt nội dung báo cáo
        preview_box = ctk.CTkFrame(container, corner_radius=14, fg_color=CARD_BG, border_width=1, border_color=BORDER_COLOR)
        preview_box.pack(fill="both", expand=True)

        ctk.CTkLabel(preview_box, text="BẢN XEM TRƯỚC TÓM TẮT TOÁN HỌC & KỸ THUẬT CỦA BÁO CÁO BTL", font=(FONT_MAIN, 10, "bold"), text_color=ACCENT_CYAN).pack(anchor="w", padx=14, pady=(10, 4))

        self.txt_preview = ctk.CTkTextbox(
            preview_box,
            fg_color=CARD_SUB_BG,
            text_color=TEXT_MAIN,
            font=("Consolas", 9),
            corner_radius=10
        )
        self.txt_preview.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._populate_report_preview()

    # ==================================================================================
    # PHƯƠNG THỨC TRỢ GIÚP ĐỒ THỊ MATPLOTLIB DARK THEME
    # ==================================================================================
    def _style_axis_dark(self, ax):
        """Áp dụng quy chuẩn thiết kế Dark Theme lên trục Matplotlib."""
        ax.set_facecolor(CARD_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(BORDER_COLOR)
        ax.spines["left"].set_color(BORDER_COLOR)
        ax.tick_params(colors=TEXT_MUTED, labelsize=7.5)
        ax.grid(True, color=CARD_SUB_BG, linestyle="--", alpha=0.5)

    # ==================================================================================
    # CÁC HÀM XỬ LÝ SỰ KIỆN & KẾT NỐI BACKEND ct_sim/
    # ==================================================================================
    def _browse_real_ct_file(self):
        """[Backend Connection: ct_sim.phantoms.register_custom_real_ct] Nạp file ảnh CT thực tế bất kỳ."""
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
        preset = self.cb_cal_preset.get()
        register_custom_real_ct(custom_key, fp, preset)

        # Cập nhật danh sách Combobox ở cả Tab 1 và Tab 4
        all_keys = list(PHANTOMS.keys())
        self.cb_phantom.configure(values=all_keys)
        self.cb_phantom.set(custom_key)

        real_keys = [k for k in all_keys if "CT Thực Tế" in k]
        self.cb_real_case.configure(values=real_keys)
        self.cb_real_case.set(custom_key)

        self._fast_scan()
        self._run_real_ct_deep_analysis(custom_key)

    def _on_phantom_selected(self):
        name = self.cb_phantom.get()
        if "Gan" in name:
            self.cb_cal_preset.set("Abdomen (Gan / Bụng)")
        elif "Não" in name or "Đầu" in name:
            self.cb_cal_preset.set("Brain (Sọ não)")
        elif "Phổi" in name or "ngực" in name:
            self.cb_cal_preset.set("Lung (Phổi HRCT)")
        self._on_param_changed()
        self._fast_scan()

    def _on_calibration_changed(self):
        if self.custom_file_path:
            fname = os.path.basename(self.custom_file_path)
            custom_key = f"CT Thực Tế: {fname}"
            register_custom_real_ct(custom_key, self.custom_file_path, self.cb_cal_preset.get())
            if self.cb_phantom.get() == custom_key:
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

    def _on_kv_changed(self, val_str: str):
        self.lbl_kv_disp.configure(text=val_str)
        self._on_param_changed()

    def _on_ma_slider(self, val: float):
        self.lbl_ma_disp.configure(text=f"{int(round(val))} mA")
        self._on_param_changed()

    def _get_metal_val(self) -> int:
        txt = self.seg_metal.get()
        if "1" in txt:
            return 1
        if "2" in txt:
            return 2
        return 0

    def _on_param_changed(self):
        """[Backend Connection: ct_sim.physics.estimate_dose] Cập nhật liều bức xạ ALARA theo thời gian thực."""
        kv_str = self.seg_kv.get().replace(" kV", "")
        kv = float(kv_str) if kv_str.isdigit() else 120.0
        ma = float(self.slider_ma.get())
        views = int(self.cb_views.get())

        _, info = make_phantom(self.cb_phantom.get(), 128)
        params = ScanParams(kv=kv, ma=ma, n_views=views)
        dose = estimate_dose(params, info)

        self.pill_ctdi.update_badge(f"{dose['CTDIvol_mGy']:.1f} mGy")
        self.pill_dlp.update_badge(f"{dose['DLP_mGycm']:.1f} mGy·cm | {dose['E_mSv']:.2f} mSv")
        self.pill_flux.update_badge(f"{dose['I0']:.2e} ph/bin")

        pct = dose["rel_dose_pct"]
        if pct <= 25.0:
            self.badge_header_alara.update_badge(f"Liều cực thấp (-{100-pct:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)", "#60a5fa", "#1e3a8a", "#2563eb")
        elif pct <= 65.0:
            self.badge_header_alara.update_badge(f"Giảm liều tốt (-{100-pct:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)", "#34d399", "#064e3b", "#059669")
        elif pct <= 100.0:
            self.badge_header_alara.update_badge(f"Chuẩn lâm sàng ({pct:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)", "#86efac", "#14532d", "#16a34a")
        else:
            self.badge_header_alara.update_badge(f"Cảnh báo: Liều cao (+{pct-100:.0f}% | {dose['CTDIvol_mGy']:.1f} mGy)", "#fca5a5", "#7f1d1d", "#dc2626")

    def _quick_initial_scan(self):
        self._on_param_changed()
        self._fast_scan()
        self._run_real_ct_deep_analysis(self.cb_real_case.get(), switch_tab=False)

    def _fast_scan(self):
        """[Backend Connection: ct_sim.physics.simulate_scan & ct_sim.pipeline.ReconContext] Quét nhanh tức thì."""
        self.lbl_scan_status.configure(text="Đang mô phỏng thu nhận & tái tạo FBP...")
        self.progress_scan.set(0.3)
        self.update_idletasks()

        name = self.cb_phantom.get()
        metal = self._get_metal_val()
        kv_str = self.seg_kv.get().replace(" kV", "")
        kv = float(kv_str) if kv_str.isdigit() else 120.0

        params = ScanParams(
            kv=kv,
            ma=float(self.slider_ma.get()),
            n_views=int(self.cb_views.get()),
            poisson=bool(self.chk_poisson.get()),
            beam_hardening=bool(self.chk_hardening.get())
        )

        t0 = time.time()
        self.current_scan = simulate_scan(name, 128, params, metal=metal)
        self.recon_context = ReconContext(self.current_scan)
        cfg = {**DEFAULT_CFG, "fbp_filter": self.cb_fbp_filter.get()}
        fbp_res = self.recon_context.run("FBP", cfg)
        self.algo_results = {"FBP": fbp_res}

        self.progress_scan.set(1.0)
        self.lbl_scan_status.configure(text=f"Hoàn tất quét & tái tạo FBP ({time.time()-t0:.2f}s).")
        self._draw_tab1_results(fbp_res.hu)

    def _redraw_current_tab1(self):
        if "FBP" in self.algo_results:
            self._draw_tab1_results(self.algo_results["FBP"].hu)

    def _draw_tab1_results(self, rec_hu: np.ndarray):
        """Hiển thị đồng bộ 3 Subplot trên Viewport Tab 1 và cập nhật Pill Badges."""
        scan = self.current_scan
        if scan is None:
            return

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self._style_axis_dark(self.ax1)
        self._style_axis_dark(self.ax2)
        self._style_axis_dark(self.ax3)

        lvl, wid = scan.info.window
        vmin, vmax = lvl - wid / 2, lvl + wid / 2

        # 1. Đối tượng chụp (Phantom / Ảnh CT Thực tế)
        self.ax1.imshow(scan.hu_true, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax1.set_title(f"1. Đối tượng: {scan.info.name}\n(W:{wid:.0f}, L:{lvl:.0f} HU)", color="#38bdf8", fontsize=9.5, fontweight="bold")
        self.ax1.axis("off")

        # Vẽ vòng tròn ROI nếu bật switch
        if self.switch_show_roi.get():
            for rname, rcol, rlbl in [("bg", "#22d3ee", "ROI Nền (bg)"), ("lesion", "#facc15", "ROI Tổn thương")]:
                cx, cy, r = scan.info.rois[rname]
                px = (cx + 1.0) * 0.5 * scan.n
                py = (1.0 - cy) * 0.5 * scan.n
                pr = r * 0.5 * scan.n
                self.ax1.add_patch(Circle((px, py), pr, fill=False, edgecolor=rcol, lw=1.8, label=rlbl))
            self.ax1.legend(loc="lower right", fontsize=7, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        # Cập nhật Pill Badges ROI & CNR
        bg_m = roi_mask(scan.n, scan.info.rois["bg"])
        le_m = roi_mask(scan.n, scan.info.rois["lesion"])
        bg_mean = float(np.mean(rec_hu[bg_m]))
        bg_std = float(np.std(rec_hu[bg_m]))
        le_mean = float(np.mean(rec_hu[le_m]))
        cnr_val = abs(le_mean - bg_mean) / max(bg_std, 1e-6)

        m = evaluate(rec_hu, scan.hu_true, scan.info)

        self.pill_roi.update_badge(f"Nền: {bg_mean:.0f}±{bg_std:.1f} HU | Đích: {le_mean:.0f} HU")
        self.pill_cnr.update_badge(f"CNR: {cnr_val:.1f} | Noise: {bg_std:.1f} HU")

        # 2. Sinogram
        self.ax2.imshow(scan.sino, cmap="inferno", aspect="auto", origin="lower",
                        extent=[0, scan.params.angular_range, -scan.n // 2, scan.n // 2])
        self.ax2.set_title(f"2. Sinogram p(s,θ) ({scan.params.n_views} góc)", color="#fb923c", fontsize=9.5, fontweight="bold")
        self.ax2.set_xlabel("Góc quét θ (độ)", color=TEXT_MUTED, fontsize=8)
        self.ax2.set_ylabel("Kênh Detector s", color=TEXT_MUTED, fontsize=8)

        # 3. FBP Reconstructed
        flt_name = self.cb_fbp_filter.get()
        self.ax3.imshow(rec_hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax3.set_title(f"3. Khôi phục FBP ({flt_name})\nPSNR: {m['PSNR_dB']:.1f} dB | Noise: {bg_std:.1f} HU", color="#4ade80", fontsize=9.5, fontweight="bold")
        self.ax3.axis("off")

        self.fig_scan.tight_layout()
        self.canvas_scan.draw()

    # ----------------------------------------------------------------------------------
    # QUÉT ĐỘNG LIVE (ANIMATION TIẾN TRÌNH TRỰC TIẾP TRÊN TAB 1)
    # ----------------------------------------------------------------------------------
    def _start_live_scan_thread(self):
        """[Backend Connection: ct_sim.recon.fbp_accumulate] Chạy luồng quét hoạt họa."""
        if self.is_scanning:
            return
        self.is_scanning = True
        self.cancel_scan = False
        self.btn_live_scan.configure(state="disabled")
        self.btn_fast_scan.configure(state="disabled")
        self.btn_stop_scan.configure(state="normal")
        threading.Thread(target=self._run_live_scan, daemon=True).start()

    def _stop_scan(self):
        self.cancel_scan = True
        self.lbl_scan_status.configure(text="Đang dừng quét...")

    def _run_live_scan(self):
        name = self.cb_phantom.get()
        metal = self._get_metal_val()
        kv_str = self.seg_kv.get().replace(" kV", "")
        kv = float(kv_str) if kv_str.isdigit() else 120.0

        params = ScanParams(
            kv=kv,
            ma=float(self.slider_ma.get()),
            n_views=int(self.cb_views.get()),
            poisson=bool(self.chk_poisson.get()),
            beam_hardening=bool(self.chk_hardening.get())
        )

        scan = simulate_scan(name, 128, params, metal=metal)
        self.current_scan = scan
        self.recon_context = ReconContext(scan)

        flt = self.cb_fbp_filter.get()
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

        self._style_axis_dark(self.ax1)
        self._style_axis_dark(self.ax2)
        self._style_axis_dark(self.ax3)

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

        self.ax3.imshow(rec_hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax3.set_title("3. Tái tạo FBP (Cộng dồn)", color="#4ade80", fontsize=10, fontweight="bold")
        self.ax3.axis("off")

        self.canvas_scan.draw_idle()
        pct = float((f_idx + 1) / total)
        self.progress_scan.set(pct)
        self.lbl_scan_status.configure(text=f"Đang quét góc θ = {angle:.0f}° ({int(pct*100)}%)...")

    def _finish_live_scan(self):
        self.is_scanning = False
        self.btn_live_scan.configure(state="normal")
        self.btn_fast_scan.configure(state="normal")
        self.btn_stop_scan.configure(state="disabled")
        self.lbl_scan_status.configure(text="Hoàn tất quét động trực tiếp.")

    # ----------------------------------------------------------------------------------
    # TAB 2: SO SÁNH ĐA THUẬT TOÁN & BÓC TÁCH 5 BỘ LỌC FBP
    # ----------------------------------------------------------------------------------
    def _start_compare_thread(self):
        if self.current_scan is None:
            self._fast_scan()
        threading.Thread(target=self._run_compare, daemon=True).start()

    def _run_compare(self):
        """[Backend Connection: ct_sim.pipeline.run_algorithms] So sánh 7 thuật toán."""
        scan = self.current_scan
        if scan is None or self.recon_context is None:
            return

        selected_keys = [k for k, chk in self.algo_vars.items() if chk.get()]
        if not selected_keys:
            selected_keys = ["FBP"]

        self.lbl_scan_status.configure(text="Đang tái tạo các thuật toán...")
        cfg = {**DEFAULT_CFG, "fbp_filter": self.cb_fbp_filter.get()}
        results = run_algorithms(scan, selected_keys, cfg, ctx=self.recon_context)
        self.algo_results = results

        self.after(0, self._update_compare_plots)
        self.after(0, self._update_metrics_table)
        self.after(0, self._update_profile_plot)

    def _get_display_window(self, scan: ScanData):
        win_str = self.cb_win.get()
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
        cmap = self.cb_cmap.get()

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
            self._style_axis_dark(ax)
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
        """[Backend Connection: ct_sim.recon.fourier_filter & fbp] Bóc tách 5 bộ lọc FBP."""
        if self.current_scan is None:
            self._fast_scan()
        scan = self.current_scan
        if scan is None:
            return

        self.fig_compare.clear()
        lvl, wid = self._get_display_window(scan)
        vmin, vmax = lvl - wid / 2, lvl + wid / 2
        cmap = self.cb_cmap.get()

        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)

        flt_colors = {"ramp": "#ef4444", "shepp-logan": "#f59e0b", "cosine": "#10b981", "hamming": "#38bdf8", "hann": "#a855f7"}
        for idx, flt in enumerate(FILTERS):
            t0 = time.perf_counter()
            hu_f = 1000.0 * (fbp(scan.sino, scan.theta, flt, self.recon_context.lin) - 1.0)
            dt = time.perf_counter() - t0
            m = evaluate(hu_f, scan.hu_true, scan.info)

            ax = self.fig_compare.add_subplot(2, 3, idx + 1)
            self._style_axis_dark(ax)
            ax.imshow(hu_f, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"FBP ({flt.upper()})\nPSNR: {m['PSNR_dB']:.1f} dB | Noise: {m['Noise_HU']:.1f} HU | CNR: {m['CNR']:.1f}",
                color=flt_colors[flt], fontsize=7.8, fontweight="bold"
            )
            ax.axis("off")

            self.tree_metrics.insert("", "end", values=(
                f"FBP ({flt})", f"{m['PSNR_dB']:.2f}", f"{m['SSIM']:.3f}",
                f"{m['Noise_HU']:.1f}", f"{m['CNR']:.2f}", f"{dt:.2f}s"
            ))

        # Ô thứ 6: Vẽ đáp ứng tần số H(ω) = |ω| W(ω) của 5 bộ lọc
        ax_f = self.fig_compare.add_subplot(2, 3, 6)
        self._style_axis_dark(ax_f)
        freqs = np.fft.rfftfreq(128) * 2.0
        for flt in FILTERS:
            h = fourier_filter(128, flt).ravel()[:len(freqs)]
            ax_f.plot(freqs, h, color=flt_colors[flt], lw=1.6, label=flt)
        ax_f.set_title("Đáp ứng Tần số H(ω) = |ω|·W(ω)", color=TEXT_MAIN, fontsize=8, fontweight="bold")
        ax_f.set_xlabel("ω / ω_Nyquist", color=TEXT_MUTED, fontsize=7.5)
        ax_f.legend(fontsize=6.5, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        self.fig_compare.tight_layout()
        self.canvas_compare.draw()

    def _update_metrics_table(self):
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)

        for k, res in self.algo_results.items():
            m = res.metrics
            self.tree_metrics.insert("", "end", values=(
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
        self._style_axis_dark(self.ax_prof)
        scan = self.current_scan
        _, cy_le, _ = scan.info.rois["lesion"]
        mid_row = int(np.clip((1.0 - cy_le) * 0.5 * scan.n, 10, scan.n - 10))

        self.ax_prof.plot(scan.hu_true[mid_row, :], color="#ffffff", lw=2, linestyle="--", label="Sự thật nền")

        colors = {"FBP": "#ef4444", "MBIR": "#10b981", "DLIR": "#38bdf8", "ASIR": "#f59e0b", "SART": "#a855f7"}
        for k in ("FBP", "SART", "MBIR", "ASIR", "DLIR"):
            if k in self.algo_results:
                res = self.algo_results[k]
                c = colors.get(k, "#94a3b8")
                self.ax_prof.plot(res.hu[mid_row, :], color=c, lw=1.3, alpha=0.9, label=k)

        self.ax_prof.set_title(f"Mặt cắt ngang qua ROI Tổn thương (Row {mid_row})", color=TEXT_MAIN, fontsize=8.5)
        self.ax_prof.set_xlabel("Vị trí Pixel x", color=TEXT_MUTED, fontsize=8)
        self.ax_prof.set_ylabel("Độ suy giảm (HU)", color=TEXT_MUTED, fontsize=8)
        self.ax_prof.legend(fontsize=7, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        self.fig_profile.tight_layout()
        self.canvas_profile.draw()

    # ----------------------------------------------------------------------------------
    # TAB 3: KHẢO SÁT NGUYÊN LÝ ALARA
    # ----------------------------------------------------------------------------------
    def _start_alara_sweep_thread(self):
        self.btn_run_alara.configure(state="disabled")
        self.lbl_alara_status.configure(text="Đang tính toán quét liều qua 5 mức mA...")
        threading.Thread(target=self._run_alara_sweep, daemon=True).start()

    def _run_alara_sweep(self):
        """[Backend Connection: ct_sim.pipeline.dose_sweep] Khảo sát suy giảm chất lượng theo mA."""
        name = self.cb_phantom.get()
        kv_str = self.seg_kv.get().replace(" kV", "")
        kv = float(kv_str) if kv_str.isdigit() else 120.0

        base_params = ScanParams(kv=kv, n_views=180, poisson=True)
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
            self._style_axis_dark(ax)

        for k in res:
            ax1.plot(res[k]["dose"], res[k]["PSNR_dB"], marker="o", color=colors[k], lw=1.8, label=k)
        ax1.set_title("1. Tỷ số Tín hiệu trên Nhiễu (PSNR vs Liều)", color=TEXT_MAIN, fontsize=9, fontweight="bold")
        ax1.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("PSNR (dB)", color=TEXT_MUTED, fontsize=8)
        ax1.legend(fontsize=8, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        for k in res:
            ax2.plot(res[k]["dose"], res[k]["SSIM"], marker="s", color=colors[k], lw=1.8, label=k)
        ax2.set_title("2. Độ Tương Đồng Cấu Trúc (SSIM vs Liều)", color=TEXT_MAIN, fontsize=9, fontweight="bold")
        ax2.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("SSIM", color=TEXT_MUTED, fontsize=8)
        ax2.legend(fontsize=8, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        for k in res:
            ax3.plot(res[k]["ma"], res[k]["Noise_HU"], marker="^", color=colors[k], lw=1.8, label=k)
        ax3.set_title("3. Mức Nhiễu Hạt ROI (Noise HU vs mA ~ 1/√mA)", color=TEXT_MAIN, fontsize=9, fontweight="bold")
        ax3.set_xlabel("Dòng bóng phát mA", color=TEXT_MUTED, fontsize=8)
        ax3.set_ylabel("Độ lệch chuẩn Nhiễu (HU)", color=TEXT_MUTED, fontsize=8)
        ax3.legend(fontsize=8, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        for k in res:
            ax4.plot(res[k]["dose"], res[k]["CNR"], marker="d", color=colors[k], lw=1.8, label=k)
        ax4.axhline(5.0, color="#facc15", linestyle="--", lw=1.0, label="Ngưỡng Rose (CNR=5)")
        ax4.set_title("4. Tương Phản trên Nhiễu (CNR vs Liều)", color=TEXT_MAIN, fontsize=9, fontweight="bold")
        ax4.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=8)
        ax4.set_ylabel("CNR", color=TEXT_MUTED, fontsize=8)
        ax4.legend(fontsize=7.5, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        self.fig_alara.tight_layout()
        self.canvas_alara.draw()

        self.btn_run_alara.configure(state="normal")
        self.lbl_alara_status.configure(text="Khảo sát hoàn tất. MBIR & DLIR vượt trội ở liều thấp (<15 mGy).")

    # ----------------------------------------------------------------------------------
    # TAB 4: PHÂN TÍCH CHUYÊN SÂU CT THỰC TẾ & FLOWCHARTS
    # ----------------------------------------------------------------------------------
    def _run_real_ct_deep_analysis(self, case_name: str, switch_tab: bool = False):
        """[Backend Connection: ct_sim.real_ct & ct_sim.pipeline.run_algorithms] Phân tích CT thực tế."""
        if switch_tab:
            self.tabview.set("🔬 4. Phân Tích CT Thực Tế & Flowcharts")

        kv_str = self.seg_kv.get().replace(" kV", "")
        kv = float(kv_str) if kv_str.isdigit() else 120.0

        params = ScanParams(kv=kv, ma=70.0, n_views=180, poisson=True)
        sc = simulate_scan(case_name, 128, params)
        self.real_ct_scan = sc
        res = run_algorithms(sc, ["FBP", "MBIR", "DLIR"])
        self.real_ct_results = res
        self._render_tab4_real_ct()

    def _on_tab4_canvas_click(self, event):
        sc = getattr(self, "real_ct_scan", None)
        if sc is None or event.xdata is None or event.ydata is None:
            return
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
            self._style_axis_dark(ax)

        # 1. Ảnh CT Thực tế gốc + Vòng tròn ROI
        ax1.imshow(sc.hu_true, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        for rname, rcol, rlbl in [("bg", "#22d3ee", "ROI Nền"), ("lesion", "#facc15", "ROI Đích")]:
            cx, cy, r = sc.info.rois[rname]
            px = (cx + 1.0) * 0.5 * sc.n
            py = (1.0 - cy) * 0.5 * sc.n
            pr = r * 0.5 * sc.n
            ax1.add_patch(Circle((px, py), pr, fill=False, edgecolor=rcol, lw=1.8, label=rlbl))
        ax1.set_title(f"1. {sc.info.name}\n(Click Trái: ROI Nền | Phải: ROI Đích)", color="#38bdf8", fontsize=8.5, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=6.5, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)
        ax1.axis("off")

        # 2. Phổ phân bố Hounsfield (Histogram HU trong cơ thể)
        body_vals = sc.hu_true[sc.hu_true > -920.0]
        ax2.hist(body_vals, bins=45, color="#38bdf8", alpha=0.8, edgecolor="#0284c7")
        ax2.axvline(0.0, color="#4ade80", linestyle="--", lw=1.2, label="Nước (0 HU)")
        ax2.set_title("2. Phân bố Mật độ Hounsfield (HU)\nPhân tích thành phần mô học", color=TEXT_MAIN, fontsize=8.5, fontweight="bold")
        ax2.set_xlabel("Giá trị Hounsfield (HU)", color=TEXT_MUTED, fontsize=7.5)
        ax2.set_ylabel("Số lượng Voxel", color=TEXT_MUTED, fontsize=7.5)
        ax2.legend(fontsize=7, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        # 3. Sinogram p(s,θ)
        ax3.imshow(sc.sino, cmap="inferno", aspect="auto", origin="lower", extent=[0, 180, -64, 64])
        ax3.set_title("3. Biến đổi Radon p(s,θ)\n(Chiếu xạ ở liều thấp 70 mA)", color="#fb923c", fontsize=8.5, fontweight="bold")
        ax3.set_xlabel("Góc θ (độ)", color=TEXT_MUTED, fontsize=7.5)

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
        self.txt_real_stats.delete("1.0", "end")
        self.txt_real_stats.insert("end", stats_txt)

    def _display_flowchart_on_tab4(self, filename: str, title_label: str):
        """[Backend Connection: report.figures] Hiển thị sơ đồ khối thuật toán độ nét cao."""
        fig_dir = os.path.join(os.path.dirname(__file__), "report", "figures")
        img_path = os.path.join(fig_dir, filename)
        if not os.path.exists(img_path):
            from report.generate_flowcharts import generate_all_flowcharts_and_deep_figs
            generate_all_flowcharts_and_deep_figs(fig_dir)

        if not os.path.exists(img_path):
            messagebox.showwarning("Thông báo", f"Không tìm thấy sơ đồ: {img_path}")
            return

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
        self.txt_preview.delete("1.0", "end")
        self.txt_preview.insert("end", txt)

    def _generate_pdf_report(self):
        """[Backend Connection: report.generate_btl_pdf.create_btl_report] Tạo file PDF BTL."""
        self.btn_gen_pdf.configure(state="disabled", text="⏳ Đang sinh PDF...")
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
                self.after(0, lambda: self.btn_gen_pdf.configure(state="normal", text="📄 TẠO BÁO CÁO BTL PDF CHUẨN BÁCH KHOA"))

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


# ======================================================================================
# ĐIỂM KHỞI CHẠY CHÍNH (ENTRY POINT)
# ======================================================================================
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
