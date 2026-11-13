"""
gui.py - Trạm Làm Việc Y Tế Chuyên Nghiệp Cao Cấp (Medical Pro Workstation)
Hệ Thống Mô Phỏng, Giải Bài Toán Ngược & Phân Tích Cắt Lớp Vi Tính (CT Simulator Pro).

Được xây dựng hoàn toàn trên nền tảng PyQt6 và qdarktheme hiện đại:
  - Hỗ trợ màn hình độ phân giải cao HiDPI (PassThrough Policy) chống nhòe (crisp rendering)
  - Khắc phục triệt để lỗi chồng đè hình ảnh (Image Overlapping Bug Fix với ax.clear())
  - Tỷ lệ chuẩn 1:3: Sidebar điều khiển (25%) & Viewport y sinh chuyên dụng (75%)
  - Hệ thống huy hiệu viên thuốc (Pill Badges) đo chỉ số liều bức xạ ALARA & CNR/Noise
  - Tích hợp 5 Tab tính năng hoàn chỉnh, bảo toàn 100% các API backend trong ct_sim/
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# 1. Kích hoạt chế độ HiDPI trước khi khởi tạo bất kỳ widget Qt nào
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFrame, QLabel, QPushButton, QSlider, QComboBox,
    QCheckBox, QRadioButton, QButtonGroup, QProgressBar, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QScrollArea, QSizePolicy
)
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage

# 2. Cấu hình Matplotlib sử dụng backend QtAgg tương thích PyQt6
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# 3. Áp dụng qdarktheme cho toàn bộ giao diện
import qdarktheme

# ======================================================================================
# KẾT NỐI CÁC MODULE CỐT LÕI TỪ BACKEND PACKAGE ct_sim/ & report/
# ======================================================================================
# Quản lý mô hình giải phẫu & ảnh CT lâm sàng thực tế
from ct_sim.phantoms import PHANTOMS, make_phantom, roi_mask, circle_mask, register_custom_real_ct
from ct_sim.real_ct import CALIBRATION_PRESETS, load_real_ct_image, ensure_builtin_real_ct_samples

# Vật lý phát tia X, liều bức xạ ALARA & Poisson noise
from ct_sim.physics import ScanParams, ScanData, simulate_scan, estimate_dose

# Thuật toán tái tạo giải tích FBP & Bộ lọc tần số
from ct_sim.recon import FILTERS, fourier_filter, fbp, fbp_accumulate

# Trình điều phối thuật toán (BP, FBP, SART, SIRT, MBIR, ASIR, DLIR) & Quét liều
from ct_sim.pipeline import ALGOS, DEFAULT_CFG, ReconContext, run_algorithms, dose_sweep, label_for, AlgoResult

# Hệ thống chỉ số đánh giá định lượng
from ct_sim.metrics import evaluate

# ======================================================================================
# THIẾT LẬP BẢNG MÀU Y TẾ CAO CẤP (MEDICAL DARK SLATE PALETTE)
# ======================================================================================
APP_BG = "#090d16"          # Deep Navy Black (Nền toàn ứng dụng)
CARD_BG = "#111827"         # Dark Slate Card Surface (Thẻ chính)
CARD_SUB_BG = "#1e293b"     # Slate 800 Surface (Thẻ con / Điều khiển)
BORDER_COLOR = "#334155"    # Viền phân cách
ACCENT_CYAN = "#06b6d4"     # Neon Medical Cyan (Màu nhấn chính)
ACCENT_CYAN_HOVER = "#22d3ee"
ACCENT_EMERALD = "#10b981"  # Xanh an toàn ALARA / Khôi phục chuẩn
ACCENT_AMBER = "#f59e0b"    # Vàng cảnh báo liều / Khối u ROI
ACCENT_ROSE = "#f43f5e"     # Đỏ tia X / Cảnh báo nguy hiểm
TEXT_MAIN = "#f8fafc"       # Chữ trắng sáng
TEXT_MUTED = "#94a3b8"      # Chữ xám mờ
FONT_FAMILY = "Segoe UI"


# ======================================================================================
# COMPONENT GIAO DIỆN PHỤ TRỢ: PILL BADGE (HUY HIỆU VIÊN THUỐC ĐO CHỈ SỐ)
# ======================================================================================
class MetricPillBadge(QFrame):
    """Huy hiệu viên thuốc bo tròn (Rounded Pill Badge) hiển thị chỉ số lâm sàng HiDPI."""
    def __init__(self, prefix: str, value: str,
                 accent_color: str = "#38bdf8", bg_color: str = CARD_SUB_BG,
                 border_color: str = BORDER_COLOR, parent=None):
        super().__init__(parent)
        self.prefix = prefix
        self.setObjectName("PillBadge")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(4)
        
        self.lbl = QLabel(f"{prefix}: {value}")
        self.lbl.setFont(QFont(FONT_FAMILY, 9, QFont.Weight.Bold))
        layout.addWidget(self.lbl)
        
        self.update_colors(accent_color, bg_color, border_color)

    def update_colors(self, accent_color: str, bg_color: str, border_color: str):
        self.setStyleSheet(f"""
            QFrame#PillBadge {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 13px;
            }}
            QLabel {{
                color: {accent_color};
                background: transparent;
                font-family: "{FONT_FAMILY}";
                font-size: 11px;
                font-weight: bold;
            }}
        """)

    def update_badge(self, value: str, accent_color: Optional[str] = None,
                     bg_color: Optional[str] = None, border_color: Optional[str] = None):
        self.lbl.setText(f"{self.prefix}: {value}")
        if accent_color or bg_color or border_color:
            cur_acc = accent_color or "#38bdf8"
            cur_bg = bg_color or CARD_SUB_BG
            cur_bd = border_color or BORDER_COLOR
            self.update_colors(cur_acc, cur_bg, cur_bd)


# ======================================================================================
# CÁC WORKER LUỒNG NỀN (QTHREAD) GIÚP GIAO DIỆN KHÔNG BAO GIỜ BỊ ĐƠ (ZERO FREEZE)
# ======================================================================================
class LiveScanWorker(QThread):
    """Luồng nền mô phỏng quét động tích phân góc quay trực tiếp."""
    frame_ready = pyqtSignal(float, float, float, np.ndarray, np.ndarray, int, int)
    scan_finished = pyqtSignal()

    def __init__(self, scan: ScanData, filter_name: str, parent=None):
        super().__init__(parent)
        self.scan = scan
        self.filter_name = filter_name
        self.is_cancelled = False

    def run(self):
        frames = fbp_accumulate(self.scan.sino, self.scan.theta, self.filter_name, n_frames=30)
        n_frames = len(frames)
        sino_disp = np.full_like(self.scan.sino, np.nan)
        col_step = max(1, self.scan.sino.shape[1] // n_frames)
        R = 128 * 0.6

        for f_idx in range(n_frames):
            if self.is_cancelled:
                break
            cur_angle = float(self.scan.theta[min(f_idx * col_step, len(self.scan.theta) - 1)])
            rad = float(np.radians(cur_angle))
            end_col = min((f_idx + 1) * col_step, self.scan.sino.shape[1])
            sino_disp[:, :end_col] = self.scan.sino[:, :end_col]
            rec_hu = 1000.0 * (frames[f_idx] - 1.0)

            self.frame_ready.emit(cur_angle, rad, R, sino_disp.copy(), rec_hu, f_idx, n_frames)
            time.sleep(0.04)

        self.scan_finished.emit()

    def cancel(self):
        self.is_cancelled = True


class CompareWorker(QThread):
    """Luồng nền thực thi so sánh đồng thời các thuật toán tái tạo."""
    finished = pyqtSignal(dict)

    def __init__(self, scan: ScanData, keys: List[str], cfg: dict, ctx: ReconContext, parent=None):
        super().__init__(parent)
        self.scan = scan
        self.keys = keys
        self.cfg = cfg
        self.ctx = ctx

    def run(self):
        results = run_algorithms(self.scan, self.keys, self.cfg, ctx=self.ctx)
        self.finished.emit(results)


class AlaraWorker(QThread):
    """Luồng nền tính toán quét liều bức xạ ALARA qua các mức dòng ống mA."""
    finished = pyqtSignal(dict, list)

    def __init__(self, name: str, kv: float, parent=None):
        super().__init__(parent)
        self.name = name
        self.kv = kv

    def run(self):
        base_params = ScanParams(kv=self.kv, n_views=180, poisson=True)
        ma_list = [10.0, 25.0, 50.0, 100.0, 200.0]
        keys = ["FBP", "MBIR", "DLIR"]
        res = dose_sweep(self.name, 128, base_params, ma_list, keys)
        self.finished.emit(res, ma_list)


class PdfWorker(QThread):
    """Luồng nền biên soạn Báo cáo BTL PDF 21 trang A4 chuẩn Bách Khoa."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, pdf_path: str, scan: Optional[ScanData], algo_results: dict, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.scan = scan
        self.algo_results = algo_results

    def run(self):
        try:
            from report.generate_btl_pdf import create_btl_report
            out = create_btl_report(self.pdf_path, self.scan, self.algo_results)
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))


# ======================================================================================
# LỚP CỬA SỔ CHÍNH: CT SIMULATOR WORKSTATION (PYQT6 & QDARKTHEME)
# ======================================================================================
class CTSimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 1. Cấu hình Cửa sổ & Trọng lượng HiDPI
        self.setWindowTitle("CT SIMULATOR PRO — MEDICAL PRO WORKSTATION (BME LAB)")
        self.resize(1440, 850)
        self.setMinimumSize(1200, 760)

        # 2. Đảm bảo kho mẫu ảnh CT Thực tế trong assets/real_ct/
        ensure_builtin_real_ct_samples()

        # 3. Trạng thái dữ liệu nội bộ
        self.current_scan: Optional[ScanData] = None
        self.recon_context: Optional[ReconContext] = None
        self.algo_results: Dict[str, AlgoResult] = {}
        self.custom_file_path: Optional[str] = None
        self.live_worker: Optional[LiveScanWorker] = None

        # 4. Thiết lập Giao diện Khung chính & Tabs
        self._init_ui()

        # 5. Khởi động quét nhanh ban đầu
        self._quick_initial_scan()

    # ----------------------------------------------------------------------------------
    # KHỞI TẠO BỐ CỤC GIAO DIỆN (HEADER + TABS)
    # ----------------------------------------------------------------------------------
    def _init_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("CentralWidget")
        central_widget.setStyleSheet(f"QWidget#CentralWidget {{ background-color: {APP_BG}; }}")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 8, 12, 8)
        root_layout.setSpacing(6)

        # 1. Thanh tiêu đề thương hiệu (Header Bar)
        header_bar = QFrame(self)
        header_bar.setObjectName("HeaderBar")
        header_bar.setStyleSheet(f"""
            QFrame#HeaderBar {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 10px;
            }}
        """)
        h_layout = QHBoxLayout(header_bar)
        h_layout.setContentsMargins(12, 6, 12, 6)

        # Logo Bách Khoa
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "bk_logo.png")
        if os.path.exists(logo_path):
            lbl_logo = QLabel()
            pix = QPixmap(logo_path).scaled(38, 38, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_logo.setPixmap(pix)
            h_layout.addWidget(lbl_logo)

        # Tiêu đề văn bản
        t_box = QVBoxLayout()
        t_box.setSpacing(1)
        lbl_title = QLabel("CT SIMULATOR PRO — MEDICAL PRO WORKSTATION")
        lbl_title.setFont(QFont(FONT_FAMILY, 12, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"color: {TEXT_MAIN};")

        lbl_sub = QLabel("ĐHQG TP.HCM • Trường Đại học Bách khoa | Hệ thống Giải Bài toán Ngược & Tái tạo Cắt lớp Vi tính HiDPI")
        lbl_sub.setFont(QFont(FONT_FAMILY, 9))
        lbl_sub.setStyleSheet("color: #38bdf8;")

        t_box.addWidget(lbl_title)
        t_box.addWidget(lbl_sub)
        h_layout.addLayout(t_box)
        h_layout.addStretch()

        # Huy hiệu ALARA Header
        self.badge_header_alara = MetricPillBadge("TRẠNG THÁI ALARA", "Chuẩn Lâm Sàng (120 mA | 27.0 mGy)",
                                                  accent_color="#34d399", bg_color="#064e3b", border_color="#059669")
        h_layout.addWidget(self.badge_header_alara)

        root_layout.addWidget(header_bar)

        # 2. Hệ Thống 5 Tab Chuyên Nghiệp (QTabWidget)
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {BORDER_COLOR};
                background: {APP_BG};
                border-radius: 10px;
            }}
            QTabBar::tab {{
                background: {CARD_SUB_BG};
                color: {TEXT_MUTED};
                padding: 8px 16px;
                font-family: "{FONT_FAMILY}";
                font-size: 11px;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 3px;
            }}
            QTabBar::tab:selected {{
                background: {ACCENT_CYAN};
                color: #000000;
            }}
            QTabBar::tab:hover:!selected {{
                background: {BORDER_COLOR};
                color: {TEXT_MAIN};
            }}
        """)

        # Tab 1: ⚡ Trạm Quét & Tái Tạo (Flagship Workstation: Tỷ lệ 1:3)
        self.tab_scan = QWidget()
        self._build_tab_scan()
        self.tabs.addTab(self.tab_scan, "⚡ 1. Quét & Tái Tạo (Workstation)")

        # Tab 2: 📊 So Sánh 7 Thuật Toán & 5 Bộ Lọc FBP
        self.tab_compare = QWidget()
        self._build_tab_compare()
        self.tabs.addTab(self.tab_compare, "📊 2. So Sánh Thuật Toán & 5 Bộ Lọc FBP")

        # Tab 3: 📉 Khảo Sát Liều Bức Xạ ALARA
        self.tab_alara = QWidget()
        self._build_tab_alara()
        self.tabs.addTab(self.tab_alara, "📉 3. Khảo Sát Liều Bức Xạ ALARA")

        # Tab 4: 🔬 Phân Tích CT Thực Tế & Flowcharts
        self.tab_real_fc = QWidget()
        self._build_tab_real_fc()
        self.tabs.addTab(self.tab_real_fc, "🔬 4. Phân Tích CT Thực Tế & Flowcharts")

        # Tab 5: 📄 Báo Cáo BTL & Xuất Dữ Liệu
        self.tab_export = QWidget()
        self._build_tab_export()
        self.tabs.addTab(self.tab_export, "📄 5. Báo Cáo BTL & Xuất Dữ Liệu")

        root_layout.addWidget(self.tabs)

    # ----------------------------------------------------------------------------------
    # TAB 1: ⚡ TRẠM QUÉT TRỰC TIẾP & NẠP ẢNH CT (WORKSTATION 1:3 RATIO)
    # ----------------------------------------------------------------------------------
    def _build_tab_scan(self):
        layout = QHBoxLayout(self.tab_scan)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ==============================================================================
        # CỘT TRÁI (LEFT SIDEBAR - 25% WIDTH): BẢNG ĐIỀU KHIỂN DẠNG CUỘN
        # ==============================================================================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
            QScrollBar:vertical {{
                background: {CARD_BG};
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR};
                border-radius: 4px;
            }}
        """)

        sidebar_content = QWidget()
        side_layout = QVBoxLayout(sidebar_content)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(10)

        # --- Card 1: Chọn Phantom hoặc Nạp Ảnh CT Thực Tế ---
        card_p = self._create_card_frame()
        cp_layout = QVBoxLayout(card_p)
        cp_layout.setSpacing(6)

        lbl_sec1 = QLabel("1. PHANTOM & ẢNH CT THỰC TẾ")
        lbl_sec1.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        lbl_sec1.setStyleSheet(f"color: {ACCENT_CYAN};")
        cp_layout.addWidget(lbl_sec1)

        self.cb_phantom = QComboBox()
        self.cb_phantom.addItems(list(PHANTOMS.keys()))
        self.cb_phantom.setCurrentText("Đầu (HU)")
        self.cb_phantom.currentTextChanged.connect(self._on_phantom_selected)
        cp_layout.addWidget(self.cb_phantom)

        btn_browse_real = QPushButton("📂 Nạp Ảnh CT Thực Tế (PNG/DCM/NPY...)")
        btn_browse_real.setFont(QFont(FONT_FAMILY, 9, QFont.Weight.Bold))
        btn_browse_real.setStyleSheet("""
            QPushButton {
                background-color: #eab308;
                color: #000000;
                border-radius: 6px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #facc15;
            }
        """)
        btn_browse_real.clicked.connect(self._browse_real_ct_file)
        cp_layout.addWidget(btn_browse_real)

        # Hiệu chuẩn HU
        f_cal = QHBoxLayout()
        f_cal.addWidget(QLabel("Hiệu chuẩn HU:"))
        self.cb_cal_preset = QComboBox()
        self.cb_cal_preset.addItems(list(CALIBRATION_PRESETS.keys()))
        self.cb_cal_preset.setCurrentText("Auto (Tự động)")
        self.cb_cal_preset.currentTextChanged.connect(self._on_calibration_changed)
        f_cal.addWidget(self.cb_cal_preset)
        cp_layout.addLayout(f_cal)

        # Switch hiện vòng ROI
        self.chk_show_roi = QCheckBox("Hiện vòng ROI (L: Nền, R: U)")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.stateChanged.connect(self._redraw_current_tab1)
        cp_layout.addWidget(self.chk_show_roi)

        # Cấy kim loại
        f_metal = QHBoxLayout()
        f_metal.addWidget(QLabel("Cấy kim loại:"))
        self.bg_metal = QButtonGroup(self)
        self.rad_metal0 = QRadioButton("Không")
        self.rad_metal1 = QRadioButton("1 Trụ")
        self.rad_metal2 = QRadioButton("2 Trụ")
        self.rad_metal0.setChecked(True)
        self.bg_metal.addButton(self.rad_metal0, 0)
        self.bg_metal.addButton(self.rad_metal1, 1)
        self.bg_metal.addButton(self.rad_metal2, 2)
        self.bg_metal.idToggled.connect(lambda: self._on_param_changed())

        f_metal.addWidget(self.rad_metal0)
        f_metal.addWidget(self.rad_metal1)
        f_metal.addWidget(self.rad_metal2)
        cp_layout.addLayout(f_metal)

        side_layout.addWidget(card_p)

        # --- Card 2: Thông Số Bóng Phát Tia X & Bộ Lọc FBP ---
        card_x = self._create_card_frame()
        cx_layout = QVBoxLayout(card_x)
        cx_layout.setSpacing(6)

        lbl_sec2 = QLabel("2. THÔNG SỐ BÓNG PHÁT & BỘ LỌC")
        lbl_sec2.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        lbl_sec2.setStyleSheet(f"color: {ACCENT_CYAN};")
        cx_layout.addWidget(lbl_sec2)

        # kVp
        f_kv_head = QHBoxLayout()
        f_kv_head.addWidget(QLabel("Điện áp đỉnh (kVp):"))
        self.lbl_kv_disp = QLabel("120 kV")
        self.lbl_kv_disp.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold;")
        f_kv_head.addWidget(self.lbl_kv_disp, alignment=Qt.AlignmentFlag.AlignRight)
        cx_layout.addLayout(f_kv_head)

        f_kv_rads = QHBoxLayout()
        self.bg_kv = QButtonGroup(self)
        for kv_val in (80, 100, 120, 140):
            rad = QRadioButton(f"{kv_val} kV")
            if kv_val == 120:
                rad.setChecked(True)
            self.bg_kv.addButton(rad, kv_val)
            f_kv_rads.addWidget(rad)
        self.bg_kv.idToggled.connect(self._on_kv_toggled)
        cx_layout.addLayout(f_kv_rads)

        # Dòng mA
        f_ma_head = QHBoxLayout()
        f_ma_head.addWidget(QLabel("Dòng bóng phát (mA):"))
        self.lbl_ma_disp = QLabel("120 mA")
        self.lbl_ma_disp.setStyleSheet("color: #38bdf8; font-weight: bold;")
        f_ma_head.addWidget(self.lbl_ma_disp, alignment=Qt.AlignmentFlag.AlignRight)
        cx_layout.addLayout(f_ma_head)

        self.slider_ma = QSlider(Qt.Orientation.Horizontal)
        self.slider_ma.setRange(10, 300)
        self.slider_ma.setValue(120)
        self.slider_ma.valueChanged.connect(self._on_ma_changed)
        cx_layout.addWidget(self.slider_ma)

        # Views & FBP Filter
        f_v_flt = QHBoxLayout()
        f_v_flt.addWidget(QLabel("Góc Views:"))
        self.cb_views = QComboBox()
        self.cb_views.addItems(["60", "90", "120", "180"])
        self.cb_views.setCurrentText("180")
        self.cb_views.currentTextChanged.connect(self._on_param_changed)
        f_v_flt.addWidget(self.cb_views)

        f_v_flt.addWidget(QLabel("Bộ lọc:"))
        self.cb_fbp_filter = QComboBox()
        self.cb_fbp_filter.addItems(list(FILTERS))
        self.cb_fbp_filter.setCurrentText("ramp")
        self.cb_fbp_filter.currentTextChanged.connect(self._fast_scan)
        f_v_flt.addWidget(self.cb_fbp_filter)
        cx_layout.addLayout(f_v_flt)

        self.chk_poisson = QCheckBox("Nhiễu lượng tử Poisson (Quantum Noise)")
        self.chk_poisson.setChecked(True)
        self.chk_poisson.stateChanged.connect(self._on_param_changed)
        cx_layout.addWidget(self.chk_poisson)

        self.chk_hardening = QCheckBox("Hiệu ứng Cứng hóa chùm tia (Beam Hardening)")
        self.chk_hardening.setChecked(False)
        self.chk_hardening.stateChanged.connect(self._on_param_changed)
        cx_layout.addWidget(self.chk_hardening)

        side_layout.addWidget(card_x)

        # --- Card 3: Nút Bấm Điều Khiển Quét & Tiến Trình ---
        card_ctrl = self._create_card_frame()
        cc_layout = QVBoxLayout(card_ctrl)
        cc_layout.setSpacing(6)

        lbl_sec3 = QLabel("3. THAO TÁC QUÉT & TÁI TẠO")
        lbl_sec3.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        lbl_sec3.setStyleSheet(f"color: {ACCENT_CYAN};")
        cc_layout.addWidget(lbl_sec3)

        # Large Primary "START SCAN" Button
        self.btn_fast_scan = QPushButton("⚡ START SCAN & RECONSTRUCT (FBP)")
        self.btn_fast_scan.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        self.btn_fast_scan.setFixedHeight(40)
        self.btn_fast_scan.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_CYAN};
                color: #000000;
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_CYAN_HOVER};
            }}
        """)
        self.btn_fast_scan.clicked.connect(self._fast_scan)
        cc_layout.addWidget(self.btn_fast_scan)

        # Secondary Button
        self.btn_live_scan = QPushButton("▶ Quét Động Trực Tiếp (Live Scan)")
        self.btn_live_scan.setFont(QFont(FONT_FAMILY, 9, QFont.Weight.Bold))
        self.btn_live_scan.setFixedHeight(32)
        self.btn_live_scan.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_EMERALD};
                color: #ffffff;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #34d399;
            }}
        """)
        self.btn_live_scan.clicked.connect(self._start_live_scan)
        cc_layout.addWidget(self.btn_live_scan)

        self.btn_stop_scan = QPushButton("⏹ Dừng Quét")
        self.btn_stop_scan.setFixedHeight(28)
        self.btn_stop_scan.setEnabled(False)
        self.btn_stop_scan.clicked.connect(self._stop_live_scan)
        cc_layout.addWidget(self.btn_stop_scan)

        self.progress_scan = QProgressBar()
        self.progress_scan.setRange(0, 100)
        self.progress_scan.setValue(0)
        self.progress_scan.setFixedHeight(8)
        self.progress_scan.setTextVisible(False)
        self.progress_scan.setStyleSheet(f"""
            QProgressBar {{
                background: {CARD_SUB_BG};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT_CYAN};
                border-radius: 4px;
            }}
        """)
        cc_layout.addWidget(self.progress_scan)

        self.lbl_scan_status = QLabel("Sẵn sàng quét.")
        self.lbl_scan_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        cc_layout.addWidget(self.lbl_scan_status)

        side_layout.addWidget(card_ctrl)
        side_layout.addStretch()

        scroll.setWidget(sidebar_content)
        layout.addWidget(scroll)

        # ==============================================================================
        # CỘT PHẢI (RIGHT VIEWPORT - 75% WIDTH): PILL BADGES & MATPLOTLIB WORKSTATION
        # ==============================================================================
        viewport = QWidget()
        v_layout = QVBoxLayout(viewport)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(8)

        # 1. Hàng Thẻ Pill Badges hiển thị chỉ số đo lường lâm sàng trên đỉnh
        pill_row = QFrame()
        pill_row.setObjectName("PillRow")
        pill_row.setStyleSheet(f"""
            QFrame#PillRow {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """)
        pr_layout = QHBoxLayout(pill_row)
        pr_layout.setContentsMargins(8, 6, 8, 6)
        pr_layout.setSpacing(6)

        self.pill_ctdi = MetricPillBadge("CTDIvol", "27.0 mGy", "#38bdf8")
        self.pill_dlp = MetricPillBadge("DLP & E", "405.0 mGy·cm | 0.85 mSv", "#4ade80")
        self.pill_flux = MetricPillBadge("Quang Thông I₀", "1.33e+06 ph/bin", "#94a3b8")
        self.pill_roi = MetricPillBadge("ROI Nền / Đích", "-- HU | -- HU", "#facc15")
        self.pill_cnr = MetricPillBadge("CNR / Noise", "-- | -- HU", "#f472b6")

        pr_layout.addWidget(self.pill_ctdi)
        pr_layout.addWidget(self.pill_dlp)
        pr_layout.addWidget(self.pill_flux)
        pr_layout.addWidget(self.pill_roi)
        pr_layout.addWidget(self.pill_cnr)
        pr_layout.addStretch()

        v_layout.addWidget(pill_row)

        # 2. Thẻ Lớn Chứa Khung Vẽ 3 Subplots Matplotlib (Card-Based Viewport, DPI=150)
        canvas_card = QFrame()
        canvas_card.setObjectName("CanvasCard")
        canvas_card.setStyleSheet(f"""
            QFrame#CanvasCard {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """)
        cc_box = QVBoxLayout(canvas_card)
        cc_box.setContentsMargins(8, 8, 8, 8)

        # Thiết lập Figure DPI=150 cho hiển thị siêu nét trên màn hình HiDPI
        self.fig_scan = Figure(figsize=(10, 5.0), dpi=150, facecolor=CARD_BG)
        self.ax1 = self.fig_scan.add_subplot(131, facecolor=CARD_BG)
        self.ax2 = self.fig_scan.add_subplot(132, facecolor=CARD_BG)
        self.ax3 = self.fig_scan.add_subplot(133, facecolor=CARD_BG)

        for ax in (self.ax1, self.ax2, self.ax3):
            self._style_axis_dark(ax)

        self.fig_scan.tight_layout()
        self.canvas_scan = FigureCanvasQTAgg(self.fig_scan)
        self.canvas_scan.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_scan.mpl_connect("button_press_event", self._on_scan_canvas_click)

        cc_box.addWidget(self.canvas_scan)
        v_layout.addWidget(canvas_card)

        layout.addWidget(viewport, stretch=1)

    # ----------------------------------------------------------------------------------
    # TAB 2: 📊 SO SÁNH 7 THUẬT TOÁN TÁI TẠO & 5 BỘ LỌC FBP
    # ----------------------------------------------------------------------------------
    def _build_tab_compare(self):
        layout = QVBoxLayout(self.tab_compare)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Thanh điều khiển phía trên
        top_bar = self._create_card_frame()
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(10, 6, 10, 6)

        lbl_a = QLabel("Thuật toán:")
        lbl_a.setFont(QFont(FONT_FAMILY, 9, QFont.Weight.Bold))
        lbl_a.setStyleSheet(f"color: {ACCENT_CYAN};")
        tb_layout.addWidget(lbl_a)

        self.algo_chks: Dict[str, QCheckBox] = {}
        for key in ("BP", "FBP", "SART", "SIRT", "MBIR", "ASIR", "DLIR"):
            chk = QCheckBox(key)
            if key in ("BP", "FBP", "SART", "MBIR", "ASIR", "DLIR"):
                chk.setChecked(True)
            self.algo_chks[key] = chk
            tb_layout.addWidget(chk)

        btn_run_cmp = QPushButton("▶ Tái Tạo & Đo Lường")
        btn_run_cmp.setStyleSheet(f"background-color: {ACCENT_CYAN}; color: #000000; font-weight: bold; border-radius: 6px; padding: 5px 12px;")
        btn_run_cmp.clicked.connect(self._run_compare)
        tb_layout.addWidget(btn_run_cmp)

        btn_flt_study = QPushButton("🔍 Bóc Tách 5 Bộ Lọc FBP")
        btn_flt_study.setStyleSheet(f"background-color: {ACCENT_AMBER}; color: #000000; font-weight: bold; border-radius: 6px; padding: 5px 12px;")
        btn_flt_study.clicked.connect(self._show_5_filters_analysis)
        tb_layout.addWidget(btn_flt_study)

        tb_layout.addStretch()

        tb_layout.addWidget(QLabel("Cửa sổ:"))
        self.cb_win = QComboBox()
        self.cb_win.addItems(["Auto (Theo Phantom)", "Brain (L:40, W:120)", "Abdomen (L:50, W:350)", "Bone (L:400, W:1500)", "Lung (L:-500, W:1500)", "Full (-1000, 1000)"])
        self.cb_win.setCurrentText("Auto (Theo Phantom)")
        self.cb_win.currentTextChanged.connect(self._update_compare_plots)
        tb_layout.addWidget(self.cb_win)

        tb_layout.addWidget(QLabel("Màu:"))
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(["bone", "gray", "inferno", "magma", "viridis"])
        self.cb_cmap.setCurrentText("bone")
        self.cb_cmap.currentTextChanged.connect(self._update_compare_plots)
        tb_layout.addWidget(self.cb_cmap)

        layout.addWidget(top_bar)

        # Thân chia 2 cột: Grid ảnh bên trái (stretch=1) + Bảng số liệu & Mặt cắt 1D bên phải (fixed 440px)
        split = QHBoxLayout()
        split.setSpacing(6)

        # Lưới ảnh
        img_card = self._create_card_frame()
        ic_layout = QVBoxLayout(img_card)
        ic_layout.setContentsMargins(6, 6, 6, 6)

        self.fig_compare = Figure(figsize=(7.6, 6.0), dpi=150, facecolor=CARD_BG)
        self.canvas_compare = FigureCanvasQTAgg(self.fig_compare)
        ic_layout.addWidget(self.canvas_compare)
        split.addWidget(img_card, stretch=1)

        # Cột phụ bên phải: Bảng số liệu định lượng + Mặt cắt 1D
        side_card = self._create_card_frame()
        side_card.setFixedWidth(440)
        sc_layout = QVBoxLayout(side_card)
        sc_layout.setContentsMargins(8, 8, 8, 8)
        sc_layout.setSpacing(6)

        lbl_tbl = QLabel("BẢNG CHỈ SỐ ĐỊNH LƯỢNG KHÔI PHỤC HÀM μ(x,y)")
        lbl_tbl.setFont(QFont(FONT_FAMILY, 9, QFont.Weight.Bold))
        lbl_tbl.setStyleSheet(f"color: {ACCENT_CYAN};")
        sc_layout.addWidget(lbl_tbl)

        self.table_metrics = QTableWidget()
        self.table_metrics.setColumnCount(6)
        self.table_metrics.setHorizontalHeaderLabels(["Thuật toán", "PSNR", "SSIM", "Nhiễu", "CNR", "Thời gian"])
        self.table_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_metrics.setStyleSheet(f"""
            QTableWidget {{
                background-color: {CARD_BG};
                color: {TEXT_MAIN};
                gridline-color: {BORDER_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
            QHeaderView::section {{
                background-color: {CARD_SUB_BG};
                color: #38bdf8;
                font-weight: bold;
                padding: 4px;
                border: 1px solid {BORDER_COLOR};
            }}
        """)
        self.table_metrics.setFixedHeight(180)
        sc_layout.addWidget(self.table_metrics)

        lbl_prof = QLabel("MẶT CẮT ĐƯỜNG NGANG 1D QUA TỔN THƯƠNG (HU PROFILE)")
        lbl_prof.setFont(QFont(FONT_FAMILY, 9, QFont.Weight.Bold))
        lbl_prof.setStyleSheet(f"color: {ACCENT_CYAN};")
        sc_layout.addWidget(lbl_prof)

        self.fig_profile = Figure(figsize=(4.3, 2.8), dpi=150, facecolor=CARD_SUB_BG)
        self.ax_prof = self.fig_profile.add_subplot(111, facecolor=CARD_BG)
        self._style_axis_dark(self.ax_prof)
        self.fig_profile.tight_layout()
        self.canvas_profile = FigureCanvasQTAgg(self.fig_profile)
        sc_layout.addWidget(self.canvas_profile)

        split.addWidget(side_card)
        layout.addLayout(split, stretch=1)

    # ----------------------------------------------------------------------------------
    # TAB 3: 📉 KHẢO SÁT LIỀU BỨC XẠ ALARA
    # ----------------------------------------------------------------------------------
    def _build_tab_alara(self):
        layout = QVBoxLayout(self.tab_alara)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_card = self._create_card_frame()
        tc_layout = QHBoxLayout(top_card)
        tc_layout.setContentsMargins(12, 8, 12, 8)

        lbl = QLabel("Khảo sát suy giảm chất lượng ảnh theo dòng bóng phát (mA) & liều hấp thụ ALARA (Kiểm chứng định luật σ ~ 1/√mA):")
        lbl.setFont(QFont(FONT_FAMILY, 9, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {ACCENT_CYAN};")
        tc_layout.addWidget(lbl)

        self.btn_run_alara = QPushButton("⚡ Chạy Quét Đồ Thị Liều (mA = 10 -> 200)")
        self.btn_run_alara.setStyleSheet(f"background-color: {ACCENT_AMBER}; color: #000000; font-weight: bold; border-radius: 6px; padding: 6px 14px;")
        self.btn_run_alara.clicked.connect(self._start_alara_sweep)
        tc_layout.addWidget(self.btn_run_alara)

        self.lbl_alara_status = QLabel("Nhấn nút để chạy khảo sát trên đối tượng hiện tại.")
        self.lbl_alara_status.setStyleSheet(f"color: {TEXT_MUTED};")
        tc_layout.addWidget(self.lbl_alara_status)
        tc_layout.addStretch()

        layout.addWidget(top_card)

        plot_card = self._create_card_frame()
        pc_layout = QVBoxLayout(plot_card)
        pc_layout.setContentsMargins(6, 6, 6, 6)

        self.fig_alara = Figure(figsize=(10, 5.8), dpi=150, facecolor=CARD_BG)
        self.canvas_alara = FigureCanvasQTAgg(self.fig_alara)
        pc_layout.addWidget(self.canvas_alara)

        layout.addWidget(plot_card, stretch=1)

    # ----------------------------------------------------------------------------------
    # TAB 4: 🔬 PHÂN TÍCH CT THỰC TẾ & FLOWCHARTS
    # ----------------------------------------------------------------------------------
    def _build_tab_real_fc(self):
        layout = QHBoxLayout(self.tab_real_fc)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Cột điều khiển bên trái (350px)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)
        scroll.setStyleSheet(f"QScrollArea {{ background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; }}")

        content = QWidget()
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(10, 10, 10, 10)
        c_layout.setSpacing(8)

        lbl1 = QLabel("1. KHO ẢNH CT LÂM SÀNG THỰC TẾ")
        lbl1.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        lbl1.setStyleSheet(f"color: {ACCENT_CYAN};")
        c_layout.addWidget(lbl1)

        real_presets = [k for k in PHANTOMS.keys() if "CT Thực Tế" in k]
        self.cb_real_case = QComboBox()
        self.cb_real_case.addItems(real_presets)
        self.cb_real_case.setCurrentText(real_presets[0] if real_presets else "CT Thực Tế: Gan Bình Thường")
        self.cb_real_case.currentTextChanged.connect(lambda v: self._run_real_ct_deep_analysis(v))
        c_layout.addWidget(self.cb_real_case)

        btn_open = QPushButton("📂 Mở File Ảnh CT Thực Tế Bất Kỳ...")
        btn_open.setStyleSheet("background-color: #eab308; color: #000000; font-weight: bold; border-radius: 6px; padding: 6px;")
        btn_open.clicked.connect(self._browse_real_ct_file)
        c_layout.addWidget(btn_open)

        btn_run = QPushButton("🔬 Phân Tích Chuyên Sâu Ca CT Này")
        btn_run.setStyleSheet(f"background-color: {ACCENT_CYAN}; color: #000000; font-weight: bold; border-radius: 6px; padding: 6px;")
        btn_run.clicked.connect(lambda: self._run_real_ct_deep_analysis(self.cb_real_case.currentText()))
        c_layout.addWidget(btn_run)

        lbl2 = QLabel("2. THỐNG KÊ MÔ HỌC & ĐỊNH LƯỢNG ROI")
        lbl2.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        lbl2.setStyleSheet(f"color: {ACCENT_CYAN};")
        c_layout.addWidget(lbl2)

        self.txt_real_stats = QTextEdit()
        self.txt_real_stats.setReadOnly(True)
        self.txt_real_stats.setFixedHeight(140)
        self.txt_real_stats.setStyleSheet(f"background-color: {CARD_SUB_BG}; color: {TEXT_MAIN}; font-family: 'Consolas'; font-size: 11px; border: 1px solid {BORDER_COLOR}; border-radius: 6px;")
        c_layout.addWidget(self.txt_real_stats)

        lbl3 = QLabel("3. SƠ ĐỒ KHỐI TOÁN HỌC (FLOWCHARTS)")
        lbl3.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        lbl3.setStyleSheet(f"color: {ACCENT_CYAN};")
        c_layout.addWidget(lbl3)

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
            btn = QPushButton(f"📐 {label_txt}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {CARD_SUB_BG};
                    color: {TEXT_MAIN};
                    text-align: left;
                    padding: 5px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {ACCENT_CYAN};
                    color: #000000;
                    font-weight: bold;
                }}
            """)
            btn.clicked.connect(lambda _, fn=fname, lt=label_txt: self._display_flowchart_on_tab4(fn, lt))
            c_layout.addWidget(btn)

        c_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Khung hiển thị bên phải (Canvas)
        right_card = self._create_card_frame()
        rc_layout = QVBoxLayout(right_card)
        rc_layout.setContentsMargins(6, 6, 6, 6)

        self.fig_real_fc = Figure(figsize=(10.2, 6.0), dpi=150, facecolor=CARD_BG)
        self.canvas_real_fc = FigureCanvasQTAgg(self.fig_real_fc)
        self.canvas_real_fc.mpl_connect("button_press_event", self._on_tab4_canvas_click)
        rc_layout.addWidget(self.canvas_real_fc)

        layout.addWidget(right_card, stretch=1)

    # ----------------------------------------------------------------------------------
    # TAB 5: 📄 BÁO CÁO BTL & XUẤT DỮ LIỆU
    # ----------------------------------------------------------------------------------
    def _build_tab_export(self):
        layout = QVBoxLayout(self.tab_export)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info_card = self._create_card_frame()
        ic_layout = QVBoxLayout(info_card)
        ic_layout.setContentsMargins(16, 12, 16, 12)

        lbl = QLabel("XUẤT BÁO CÁO BÀI TẬP LỚN PDF CHUẨN ĐẠI HỌC BÁCH KHOA (KÈM 6 FLOWCHARTS & TOÁN HỌC CHUYÊN SÂU)")
        lbl.setFont(QFont(FONT_FAMILY, 11, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {ACCENT_CYAN};")
        ic_layout.addWidget(lbl)

        desc = QLabel(
            "Hệ thống tự động kết xuất toàn bộ báo cáo BTL định dạng PDF chuẩn Bách Khoa (bám sát mẫu BTLXSTK_Khoa Cơ Khí_L01.pdf), "
            "bao gồm: Khung lý thuyết Bài toán ngược đặt không chỉnh (Ill-posed Inverse Problem), chứng minh hàm truyền điểm 1/(πr) của BP, "
            "bóc tách nhân Ram-Lak & định lý truyền phương sai nhiễu của 5 bộ lọc FBP, mô hình dải hình thang & bán hội tụ GSVD của SART/SIRT, "
            "đạo hàm Euler-Lagrange & giải thuật Lagged Diffusivity + PCG của MBIR, điều khiển phổ NPS của ASIR, mạng Residual CNN của DLIR, "
            "6 Flowchart thuật toán và kết quả phân tích Ảnh CT Lâm sàng Thực tế."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        ic_layout.addWidget(desc)

        btn_bar = QHBoxLayout()
        self.btn_gen_pdf = QPushButton("📄 TẠO BÁO CÁO BTL PDF CHUẨN BÁCH KHOA")
        self.btn_gen_pdf.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        self.btn_gen_pdf.setStyleSheet(f"background-color: {ACCENT_EMERALD}; color: #ffffff; border-radius: 6px; padding: 8px 16px;")
        self.btn_gen_pdf.clicked.connect(self._generate_pdf_report)
        btn_bar.addWidget(self.btn_gen_pdf)

        btn_save_img = QPushButton("💾 Lưu Toàn Bộ Ảnh Tái Tạo (PNG)")
        btn_save_img.setStyleSheet(f"background-color: {CARD_SUB_BG}; color: {TEXT_MAIN}; border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 8px 14px;")
        btn_save_img.clicked.connect(self._save_images)
        btn_bar.addWidget(btn_save_img)

        btn_csv = QPushButton("📊 Xuất Bảng Số Liệu (CSV)")
        btn_csv.setStyleSheet(f"background-color: {CARD_SUB_BG}; color: {TEXT_MAIN}; border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 8px 14px;")
        btn_csv.clicked.connect(self._export_csv)
        btn_bar.addWidget(btn_csv)

        btn_bar.addStretch()
        ic_layout.addLayout(btn_bar)

        layout.addWidget(info_card)

        # Hộp văn bản tóm tắt
        preview_card = self._create_card_frame()
        pr_layout = QVBoxLayout(preview_card)
        pr_layout.setContentsMargins(12, 10, 12, 10)

        lbl_pr = QLabel("BẢN XEM TRƯỚC TÓM TẮT TOÁN HỌC & KỸ THUẬT CỦA BÁO CÁO BTL")
        lbl_pr.setFont(QFont(FONT_FAMILY, 10, QFont.Weight.Bold))
        lbl_pr.setStyleSheet(f"color: {ACCENT_CYAN};")
        pr_layout.addWidget(lbl_pr)

        self.txt_preview = QTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setStyleSheet(f"background-color: {CARD_SUB_BG}; color: {TEXT_MAIN}; font-family: 'Consolas'; font-size: 11px; border: 1px solid {BORDER_COLOR}; border-radius: 6px;")
        pr_layout.addWidget(self.txt_preview)
        self._populate_report_preview()

        layout.addWidget(preview_card, stretch=1)

    # ----------------------------------------------------------------------------------
    # PHƯƠNG THỨC TRỢ GIÚP GIAO DIỆN & MATPLOTLIB
    # ----------------------------------------------------------------------------------
    def _create_card_frame(self) -> QFrame:
        card = QFrame()
        card.setObjectName("CardFrame")
        card.setStyleSheet(f"""
            QFrame#CardFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """)
        return card

    def _style_axis_dark(self, ax):
        """Áp dụng Dark Theme y sinh cho các trục Matplotlib."""
        ax.set_facecolor(CARD_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(BORDER_COLOR)
        ax.spines["left"].set_color(BORDER_COLOR)
        ax.tick_params(colors=TEXT_MUTED, labelsize=7)
        ax.grid(True, color=CARD_SUB_BG, linestyle="--", alpha=0.5)

    # ==================================================================================
    # LOGIC ĐIỀU KHIỂN & KẾT NỐI BACKEND ct_sim/
    # ==================================================================================
    def _on_kv_toggled(self, kv_val: int, checked: bool):
        if checked:
            self.lbl_kv_disp.setText(f"{kv_val} kV")
            self._on_param_changed()

    def _on_ma_changed(self, val: int):
        self.lbl_ma_disp.setText(f"{val} mA")
        self._on_param_changed()

    def _get_metal_val(self) -> int:
        return self.bg_metal.checkedId()

    def _on_param_changed(self):
        """[Backend Connection: ct_sim.physics.estimate_dose] Cập nhật liều ALARA theo thông số mới."""
        kv = float(self.bg_kv.checkedId()) if self.bg_kv.checkedId() > 0 else 120.0
        ma = float(self.slider_ma.value())
        views = int(self.cb_views.currentText())

        _, info = make_phantom(self.cb_phantom.currentText(), 128)
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

    def _browse_real_ct_file(self):
        """[Backend Connection: ct_sim.phantoms.register_custom_real_ct] Nạp file ảnh CT thực tế bất kỳ."""
        filetypes = "Tất cả định dạng (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.npy *.npz *.dcm);;Ảnh Y khoa (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;Mảng Khoa học & DICOM (*.npy *.npz *.dcm);;Tất cả file (*.*)"
        fp, _ = QFileDialog.getOpenFileName(self, "Chọn tập tin Ảnh CT Thực Tế để phân tích", "", filetypes)
        if not fp:
            return

        self.custom_file_path = fp
        fname = os.path.basename(fp)
        custom_key = f"CT Thực Tế: {fname}"
        preset = self.cb_cal_preset.currentText()
        register_custom_real_ct(custom_key, fp, preset)

        all_keys = list(PHANTOMS.keys())
        self.cb_phantom.clear()
        self.cb_phantom.addItems(all_keys)
        self.cb_phantom.setCurrentText(custom_key)

        real_keys = [k for k in all_keys if "CT Thực Tế" in k]
        self.cb_real_case.clear()
        self.cb_real_case.addItems(real_keys)
        self.cb_real_case.setCurrentText(custom_key)

        self._fast_scan()
        self._run_real_ct_deep_analysis(custom_key)

    def _on_phantom_selected(self, name: str):
        if "Gan" in name:
            self.cb_cal_preset.setCurrentText("Abdomen (Gan / Bụng)")
        elif "Não" in name or "Đầu" in name:
            self.cb_cal_preset.setCurrentText("Brain (Sọ não)")
        elif "Phổi" in name or "ngực" in name:
            self.cb_cal_preset.setCurrentText("Lung (Phổi HRCT)")
        self._on_param_changed()
        self._fast_scan()

    def _on_calibration_changed(self, preset: str):
        if self.custom_file_path:
            fname = os.path.basename(self.custom_file_path)
            custom_key = f"CT Thực Tế: {fname}"
            register_custom_real_ct(custom_key, self.custom_file_path, preset)
            if self.cb_phantom.currentText() == custom_key:
                self._fast_scan()

    def _quick_initial_scan(self):
        self._on_param_changed()
        self._fast_scan()
        self._run_real_ct_deep_analysis(self.cb_real_case.currentText(), switch_tab=False)

    def _fast_scan(self):
        """[Backend Connection: ct_sim.physics.simulate_scan & ct_sim.pipeline.ReconContext] Quét nhanh tức thì."""
        self.lbl_scan_status.setText("Đang mô phỏng thu nhận & tái tạo FBP...")
        self.progress_scan.setValue(30)
        QApplication.processEvents()

        name = self.cb_phantom.currentText()
        metal = self._get_metal_val()
        kv = float(self.bg_kv.checkedId()) if self.bg_kv.checkedId() > 0 else 120.0

        params = ScanParams(
            kv=kv,
            ma=float(self.slider_ma.value()),
            n_views=int(self.cb_views.currentText()),
            poisson=self.chk_poisson.isChecked(),
            beam_hardening=self.chk_hardening.isChecked()
        )

        t0 = time.time()
        self.current_scan = simulate_scan(name, 128, params, metal=metal)
        self.recon_context = ReconContext(self.current_scan)
        cfg = {**DEFAULT_CFG, "fbp_filter": self.cb_fbp_filter.currentText()}
        fbp_res = self.recon_context.run("FBP", cfg)
        self.algo_results = {"FBP": fbp_res}

        self.progress_scan.setValue(100)
        self.lbl_scan_status.setText(f"Hoàn tất quét & tái tạo FBP ({time.time()-t0:.2f}s).")
        self._draw_tab1_results(fbp_res.hu)

    def _redraw_current_tab1(self):
        if "FBP" in self.algo_results:
            self._draw_tab1_results(self.algo_results["FBP"].hu)

    def _draw_tab1_results(self, rec_hu: np.ndarray):
        """
        [CRITICAL BUG FIX]:
        Mỗi lần vẽ lại, ax.clear() được gọi tường minh trên cả 3 trục trước khi imshow,
        sau đó tái lập các thuộc tính giao diện và gọi canvas.draw().
        Tuyệt đối không xảy ra hiện tượng chồng đè hình ảnh cũ!
        """
        scan = self.current_scan
        if scan is None:
            return

        # 1. Triệt tiêu toàn bộ nội dung cũ (Clear Axes)
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self._style_axis_dark(self.ax1)
        self._style_axis_dark(self.ax2)
        self._style_axis_dark(self.ax3)

        lvl, wid = scan.info.window
        vmin, vmax = lvl - wid / 2, lvl + wid / 2

        # Subplot 1: Đối tượng gốc & Vòng tròn ROI
        self.ax1.imshow(scan.hu_true, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax1.set_title(f"1. Đối tượng: {scan.info.name}\n(W:{wid:.0f}, L:{lvl:.0f} HU)", color="#38bdf8", fontsize=8.5, fontweight="bold")
        self.ax1.axis("off")

        if self.chk_show_roi.isChecked():
            for rname, rcol, rlbl in [("bg", "#22d3ee", "ROI Nền (bg)"), ("lesion", "#facc15", "ROI Tổn thương")]:
                cx, cy, r = scan.info.rois[rname]
                px = (cx + 1.0) * 0.5 * scan.n
                py = (1.0 - cy) * 0.5 * scan.n
                pr = r * 0.5 * scan.n
                self.ax1.add_patch(Circle((px, py), pr, fill=False, edgecolor=rcol, lw=1.8, label=rlbl))
            self.ax1.legend(loc="lower right", fontsize=6.5, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        # Cập nhật Pill Badges
        bg_m = roi_mask(scan.n, scan.info.rois["bg"])
        le_m = roi_mask(scan.n, scan.info.rois["lesion"])
        bg_mean = float(np.mean(rec_hu[bg_m]))
        bg_std = float(np.std(rec_hu[bg_m]))
        le_mean = float(np.mean(rec_hu[le_m]))
        cnr_val = abs(le_mean - bg_mean) / max(bg_std, 1e-6)

        m = evaluate(rec_hu, scan.hu_true, scan.info)
        self.pill_roi.update_badge(f"Nền: {bg_mean:.0f}±{bg_std:.1f} HU | Đích: {le_mean:.0f} HU")
        self.pill_cnr.update_badge(f"CNR: {cnr_val:.1f} | Noise: {bg_std:.1f} HU")

        # Subplot 2: Sinogram
        self.ax2.imshow(scan.sino, cmap="inferno", aspect="auto", origin="lower",
                        extent=[0, scan.params.angular_range, -scan.n // 2, scan.n // 2])
        self.ax2.set_title(f"2. Sinogram p(s,θ) ({scan.params.n_views} góc)", color="#fb923c", fontsize=8.5, fontweight="bold")
        self.ax2.set_xlabel("Góc quét θ (độ)", color=TEXT_MUTED, fontsize=7)
        self.ax2.set_ylabel("Kênh Detector s", color=TEXT_MUTED, fontsize=7)

        # Subplot 3: FBP Reconstructed
        flt_name = self.cb_fbp_filter.currentText()
        self.ax3.imshow(rec_hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax3.set_title(f"3. Khôi phục FBP ({flt_name})\nPSNR: {m['PSNR_dB']:.1f} dB | Noise: {bg_std:.1f} HU", color="#4ade80", fontsize=8.5, fontweight="bold")
        self.ax3.axis("off")

        self.fig_scan.tight_layout()
        self.canvas_scan.draw()

    def _on_scan_canvas_click(self, event):
        """Cho phép nhấp Chuột Trái (đặt ROI Nền bg) hoặc Chuột Phải (đặt ROI Tổn thương lesion) trên ảnh."""
        if self.current_scan is None or event.inaxes not in (self.ax1, self.ax3):
            return
        if event.xdata is None or event.ydata is None:
            return
        n = self.current_scan.n
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

        for k, res in list(self.algo_results.items()):
            res.metrics = evaluate(res.hu, self.current_scan.hu_true, self.current_scan.info)
        self._redraw_current_tab1()
        self._update_metrics_table()

    # ----------------------------------------------------------------------------------
    # QUÉT ĐỘNG LIVE SCAN (QTHREAD KHÔNG LÀM ĐƠ GIAO DIỆN)
    # ----------------------------------------------------------------------------------
    def _start_live_scan(self):
        if self.current_scan is None:
            self._fast_scan()
        if self.live_worker and self.live_worker.isRunning():
            return

        self.btn_live_scan.setEnabled(False)
        self.btn_fast_scan.setEnabled(False)
        self.btn_stop_scan.setEnabled(True)

        self.live_worker = LiveScanWorker(self.current_scan, self.cb_fbp_filter.currentText(), self)
        self.live_worker.frame_ready.connect(self._render_live_frame)
        self.live_worker.scan_finished.connect(self._finish_live_scan)
        self.live_worker.start()

    def _stop_live_scan(self):
        if self.live_worker:
            self.live_worker.cancel()
            self.lbl_scan_status.setText("Đang dừng quét...")

    def _render_live_frame(self, angle: float, rad: float, R: float, sino_matrix: np.ndarray, rec_hu: np.ndarray, f_idx: int, total: int):
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
        self.ax1.plot(src[0], src[1], marker="o", markersize=7, mfc="#eab308", mec="#ffffff")
        self.ax1.set_title(f"1. Thu nhận: θ = {angle:.0f}°", color="#38bdf8", fontsize=9, fontweight="bold")
        self.ax1.axis("off")

        self.ax2.imshow(sino_matrix, cmap="inferno", aspect="auto", origin="lower",
                        extent=[0, scan.params.angular_range, -scan.n // 2, scan.n // 2],
                        vmin=0, vmax=scan.sino.max())
        self.ax2.axvline(x=angle, color="#38bdf8", linestyle="--", lw=1.5)
        self.ax2.set_title(f"2. Sinogram: {f_idx+1}/{total}", color="#fb923c", fontsize=9, fontweight="bold")

        self.ax3.imshow(rec_hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        self.ax3.set_title("3. Tái tạo FBP (Cộng dồn)", color="#4ade80", fontsize=9, fontweight="bold")
        self.ax3.axis("off")

        self.canvas_scan.draw()
        pct = int((f_idx + 1) / total * 100)
        self.progress_scan.setValue(pct)
        self.lbl_scan_status.setText(f"Đang quét góc θ = {angle:.0f}° ({pct}%)...")

    def _finish_live_scan(self):
        self.btn_live_scan.setEnabled(True)
        self.btn_fast_scan.setEnabled(True)
        self.btn_stop_scan.setEnabled(False)
        self.lbl_scan_status.setText("Hoàn tất quét động trực tiếp.")

    # ----------------------------------------------------------------------------------
    # TAB 2: SO SÁNH ĐA THUẬT TOÁN & BÓC TÁCH 5 BỘ LỌC FBP
    # ----------------------------------------------------------------------------------
    def _run_compare(self):
        if self.current_scan is None:
            self._fast_scan()
        scan = self.current_scan
        if scan is None or self.recon_context is None:
            return

        selected_keys = [k for k, chk in self.algo_chks.items() if chk.isChecked()]
        if not selected_keys:
            selected_keys = ["FBP"]

        self.lbl_scan_status.setText("Đang tái tạo các thuật toán...")
        cfg = {**DEFAULT_CFG, "fbp_filter": self.cb_fbp_filter.currentText()}

        self.cmp_worker = CompareWorker(scan, selected_keys, cfg, self.recon_context, self)
        self.cmp_worker.finished.connect(self._on_compare_finished)
        self.cmp_worker.start()

    def _on_compare_finished(self, results: dict):
        self.algo_results = results
        self._update_compare_plots()
        self._update_metrics_table()
        self._update_profile_plot()
        self.lbl_scan_status.setText(f"Đã tái tạo xong {len(results)} thuật toán.")

    def _get_display_window(self, scan: ScanData):
        win_str = self.cb_win.currentText()
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
        cmap = self.cb_cmap.currentText()

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
                color=title_color, fontsize=7.5, fontweight="bold"
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
        cmap = self.cb_cmap.currentText()

        self.table_metrics.setRowCount(0)
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
                color=flt_colors[flt], fontsize=7.2, fontweight="bold"
            )
            ax.axis("off")

            row = self.table_metrics.rowCount()
            self.table_metrics.insertRow(row)
            self.table_metrics.setItem(row, 0, QTableWidgetItem(f"FBP ({flt})"))
            self.table_metrics.setItem(row, 1, QTableWidgetItem(f"{m['PSNR_dB']:.2f}"))
            self.table_metrics.setItem(row, 2, QTableWidgetItem(f"{m['SSIM']:.3f}"))
            self.table_metrics.setItem(row, 3, QTableWidgetItem(f"{m['Noise_HU']:.1f}"))
            self.table_metrics.setItem(row, 4, QTableWidgetItem(f"{m['CNR']:.2f}"))
            self.table_metrics.setItem(row, 5, QTableWidgetItem(f"{dt:.2f}s"))

        # Ô thứ 6: Vẽ đáp ứng tần số H(ω) = |ω| W(ω)
        ax_f = self.fig_compare.add_subplot(2, 3, 6)
        self._style_axis_dark(ax_f)
        freqs = np.fft.rfftfreq(128) * 2.0
        for flt in FILTERS:
            h = fourier_filter(128, flt).ravel()[:len(freqs)]
            ax_f.plot(freqs, h, color=flt_colors[flt], lw=1.5, label=flt)
        ax_f.set_title("Đáp ứng Tần số H(ω) = |ω|·W(ω)", color=TEXT_MAIN, fontsize=8, fontweight="bold")
        ax_f.set_xlabel("ω / ω_Nyquist", color=TEXT_MUTED, fontsize=7)
        ax_f.legend(fontsize=6, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        self.fig_compare.tight_layout()
        self.canvas_compare.draw()

    def _update_metrics_table(self):
        self.table_metrics.setRowCount(0)
        for k, res in self.algo_results.items():
            m = res.metrics
            row = self.table_metrics.rowCount()
            self.table_metrics.insertRow(row)
            self.table_metrics.setItem(row, 0, QTableWidgetItem(res.label))
            self.table_metrics.setItem(row, 1, QTableWidgetItem(f"{m['PSNR_dB']:.2f}"))
            self.table_metrics.setItem(row, 2, QTableWidgetItem(f"{m['SSIM']:.3f}"))
            self.table_metrics.setItem(row, 3, QTableWidgetItem(f"{m['Noise_HU']:.1f}"))
            self.table_metrics.setItem(row, 4, QTableWidgetItem(f"{m['CNR']:.2f}"))
            self.table_metrics.setItem(row, 5, QTableWidgetItem(f"{res.seconds:.2f}s"))

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
                self.ax_prof.plot(res.hu[mid_row, :], color=c, lw=1.2, alpha=0.9, label=k)

        self.ax_prof.set_title(f"Mặt cắt ngang qua ROI Tổn thương (Row {mid_row})", color=TEXT_MAIN, fontsize=8)
        self.ax_prof.set_xlabel("Vị trí Pixel x", color=TEXT_MUTED, fontsize=7)
        self.ax_prof.set_ylabel("Độ suy giảm (HU)", color=TEXT_MUTED, fontsize=7)
        self.ax_prof.legend(fontsize=6.5, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        self.fig_profile.tight_layout()
        self.canvas_profile.draw()

    # ----------------------------------------------------------------------------------
    # TAB 3: KHẢO SÁT NGUYÊN LÝ ALARA
    # ----------------------------------------------------------------------------------
    def _start_alara_sweep(self):
        name = self.cb_phantom.currentText()
        kv = float(self.bg_kv.checkedId()) if self.bg_kv.checkedId() > 0 else 120.0
        self.btn_run_alara.setEnabled(False)
        self.lbl_alara_status.setText("Đang tính toán quét liều qua 5 mức mA...")

        self.alara_worker = AlaraWorker(name, kv, self)
        self.alara_worker.finished.connect(self._render_alara_plots)
        self.alara_worker.start()

    def _render_alara_plots(self, res: dict, ma_list: list):
        self.fig_alara.clear()
        colors = {"FBP": "#ef4444", "MBIR": "#10b981", "DLIR": "#38bdf8"}

        ax1 = self.fig_alara.add_subplot(221)
        ax2 = self.fig_alara.add_subplot(222)
        ax3 = self.fig_alara.add_subplot(223)
        ax4 = self.fig_alara.add_subplot(224)

        for ax in (ax1, ax2, ax3, ax4):
            self._style_axis_dark(ax)

        for k in res:
            ax1.plot(res[k]["dose"], res[k]["PSNR_dB"], marker="o", color=colors[k], lw=1.6, label=k)
        ax1.set_title("1. Tỷ số Tín hiệu trên Nhiễu (PSNR vs Liều)", color=TEXT_MAIN, fontsize=8.5, fontweight="bold")
        ax1.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=7.5)
        ax1.set_ylabel("PSNR (dB)", color=TEXT_MUTED, fontsize=7.5)
        ax1.legend(fontsize=7, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        for k in res:
            ax2.plot(res[k]["dose"], res[k]["SSIM"], marker="s", color=colors[k], lw=1.6, label=k)
        ax2.set_title("2. Độ Tương Đồng Cấu Trúc (SSIM vs Liều)", color=TEXT_MAIN, fontsize=8.5, fontweight="bold")
        ax2.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=7.5)
        ax2.set_ylabel("SSIM", color=TEXT_MUTED, fontsize=7.5)
        ax2.legend(fontsize=7, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        for k in res:
            ax3.plot(res[k]["ma"], res[k]["Noise_HU"], marker="^", color=colors[k], lw=1.6, label=k)
        ax3.set_title("3. Mức Nhiễu Hạt ROI (Noise HU vs mA ~ 1/√mA)", color=TEXT_MAIN, fontsize=8.5, fontweight="bold")
        ax3.set_xlabel("Dòng bóng phát mA", color=TEXT_MUTED, fontsize=7.5)
        ax3.set_ylabel("Độ lệch chuẩn Nhiễu (HU)", color=TEXT_MUTED, fontsize=7.5)
        ax3.legend(fontsize=7, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        for k in res:
            ax4.plot(res[k]["dose"], res[k]["CNR"], marker="d", color=colors[k], lw=1.6, label=k)
        ax4.axhline(5.0, color="#facc15", linestyle="--", lw=1.0, label="Ngưỡng Rose (CNR=5)")
        ax4.set_title("4. Tương Phản trên Nhiễu (CNR vs Liều)", color=TEXT_MAIN, fontsize=8.5, fontweight="bold")
        ax4.set_xlabel("Liều CTDIvol (mGy)", color=TEXT_MUTED, fontsize=7.5)
        ax4.set_ylabel("CNR", color=TEXT_MUTED, fontsize=7.5)
        ax4.legend(fontsize=6.5, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        self.fig_alara.tight_layout()
        self.canvas_alara.draw()

        self.btn_run_alara.setEnabled(True)
        self.lbl_alara_status.setText("Khảo sát hoàn tất. MBIR & DLIR vượt trội ở liều thấp (<15 mGy).")

    # ----------------------------------------------------------------------------------
    # TAB 4: PHÂN TÍCH CHUYÊN SÂU CT THỰC TẾ & FLOWCHARTS
    # ----------------------------------------------------------------------------------
    def _run_real_ct_deep_analysis(self, case_name: str, switch_tab: bool = False):
        if switch_tab:
            self.tabs.setCurrentIndex(3)

        kv = float(self.bg_kv.checkedId()) if self.bg_kv.checkedId() > 0 else 120.0
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
        ax1.set_title(f"1. {sc.info.name}\n(Click Trái: ROI Nền | Phải: ROI Đích)", color="#38bdf8", fontsize=8, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=6, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)
        ax1.axis("off")

        # 2. Phổ phân bố Hounsfield (Histogram HU trong cơ thể)
        body_vals = sc.hu_true[sc.hu_true > -920.0]
        ax2.hist(body_vals, bins=45, color="#38bdf8", alpha=0.8, edgecolor="#0284c7")
        ax2.axvline(0.0, color="#4ade80", linestyle="--", lw=1.2, label="Nước (0 HU)")
        ax2.set_title("2. Phân bố Mật độ Hounsfield (HU)\nPhân tích thành phần mô học", color=TEXT_MAIN, fontsize=8, fontweight="bold")
        ax2.set_xlabel("Giá trị Hounsfield (HU)", color=TEXT_MUTED, fontsize=7)
        ax2.set_ylabel("Số lượng Voxel", color=TEXT_MUTED, fontsize=7)
        ax2.legend(fontsize=6.5, facecolor=CARD_SUB_BG, labelcolor=TEXT_MAIN)

        # 3. Sinogram p(s,θ)
        ax3.imshow(sc.sino, cmap="inferno", aspect="auto", origin="lower", extent=[0, 180, -64, 64])
        ax3.set_title("3. Biến đổi Radon p(s,θ)\n(Chiếu xạ ở liều thấp 70 mA)", color="#fb923c", fontsize=8, fontweight="bold")
        ax3.set_xlabel("Góc θ (độ)", color=TEXT_MUTED, fontsize=7)

        # 4. FBP (Ramp)
        r_fbp = res["FBP"]
        ax4.imshow(r_fbp.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        ax4.set_title(f"4. Khôi phục FBP (Ramp)\nPSNR: {r_fbp.metrics['PSNR_dB']:.1f} dB | Noise: {r_fbp.metrics['Noise_HU']:.1f} HU",
                      color="#ef4444", fontsize=8, fontweight="bold")
        ax4.axis("off")

        # 5. MBIR (PWLS-TV)
        r_mbir = res["MBIR"]
        ax5.imshow(r_mbir.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        ax5.set_title(f"5. Khôi phục MBIR (PWLS-TV)\nPSNR: {r_mbir.metrics['PSNR_dB']:.1f} dB | Noise: {r_mbir.metrics['Noise_HU']:.1f} HU",
                      color="#10b981", fontsize=8, fontweight="bold")
        ax5.axis("off")

        # 6. DLIR (Residual CNN)
        r_dl = res["DLIR"]
        ax6.imshow(r_dl.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        ax6.set_title(f"6. Khôi phục DLIR (Residual CNN)\nPSNR: {r_dl.metrics['PSNR_dB']:.1f} dB | Noise: {r_dl.metrics['Noise_HU']:.1f} HU",
                      color="#38bdf8", fontsize=8, fontweight="bold")
        ax6.axis("off")

        self.fig_real_fc.tight_layout()
        self.canvas_real_fc.draw()

        # Cập nhật thống kê văn bản
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
        self.txt_real_stats.setText(stats_txt)

    def _display_flowchart_on_tab4(self, filename: str, title_label: str):
        """Hiển thị sơ đồ khối thuật toán Flowchart độ phân giải cao."""
        fig_dir = os.path.join(os.path.dirname(__file__), "report", "figures")
        img_path = os.path.join(fig_dir, filename)
        if not os.path.exists(img_path):
            from report.generate_flowcharts import generate_all_flowcharts_and_deep_figs
            generate_all_flowcharts_and_deep_figs(fig_dir)

        if not os.path.exists(img_path):
            QMessageBox.warning(self, "Thông báo", f"Không tìm thấy sơ đồ: {img_path}")
            return

        im = Image.open(img_path)
        self.fig_real_fc.clear()
        ax = self.fig_real_fc.add_subplot(111)
        ax.set_facecolor("#ffffff")
        ax.imshow(np.asarray(im))
        ax.set_title(title_label, color=ACCENT_CYAN, fontsize=9.5, fontweight="bold", pad=8)
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
        self.txt_preview.setText(txt)

    def _generate_pdf_report(self):
        """[Backend Connection: report.generate_btl_pdf.create_btl_report] Tạo Báo cáo BTL PDF."""
        self.btn_gen_pdf.setEnabled(False)
        self.btn_gen_pdf.setText("⏳ Đang sinh PDF...")

        pdf_path = os.path.join(os.path.dirname(__file__), "report", "BaoCao_BTL_Project2_CT_Scanner.pdf")
        self.pdf_worker = PdfWorker(pdf_path, self.current_scan, self.algo_results, self)
        self.pdf_worker.finished.connect(self._on_pdf_success)
        self.pdf_worker.error.connect(self._on_pdf_error)
        self.pdf_worker.start()

    def _on_pdf_success(self, out_path: str):
        self.btn_gen_pdf.setEnabled(True)
        self.btn_gen_pdf.setText("📄 TẠO BÁO CÁO BTL PDF CHUẨN BÁCH KHOA")
        QMessageBox.information(
            self,
            "Thành công",
            f"Đã tạo Báo cáo BTL PDF chuẩn Đại học Bách Khoa (kèm đầy đủ 6 Flowchart & phân tích Ảnh CT Thực Tế)!\n\nĐường dẫn:\n{out_path}"
        )

    def _on_pdf_error(self, err_msg: str):
        self.btn_gen_pdf.setEnabled(True)
        self.btn_gen_pdf.setText("📄 TẠO BÁO CÁO BTL PDF CHUẨN BÁCH KHOA")
        QMessageBox.critical(self, "Lỗi tạo PDF", f"Không thể tạo PDF:\n{err_msg}")

    def _save_images(self):
        if not self.algo_results:
            QMessageBox.warning(self, "Thông báo", "Vui lòng chạy tái tạo trước khi lưu ảnh.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh")
        if not folder:
            return

        for k, res in self.algo_results.items():
            fn = os.path.join(folder, f"CT_Recon_{k}.png")
            plt.imsave(fn, res.hu, cmap="bone")
        QMessageBox.information(self, "Thành công", f"Đã lưu {len(self.algo_results)} ảnh vào thư mục:\n{folder}")

    def _export_csv(self):
        if not self.algo_results:
            QMessageBox.warning(self, "Thông báo", "Chưa có kết quả số liệu để xuất CSV.")
            return

        fn, _ = QFileDialog.getSaveFileName(self, "Lưu số liệu CSV", "CT_Recon_Metrics.csv", "CSV Files (*.csv)")
        if not fn:
            return

        with open(fn, "w", encoding="utf-8-sig") as f:
            f.write("Thuật toán,Nhóm,PSNR (dB),SSIM,Nhiễu ROI (HU),Bias (HU),Độ tương phản (HU),CNR,Thời gian (s)\n")
            for k, res in self.algo_results.items():
                m = res.metrics
                f.write(f'"{res.label}","{res.group}",{m["PSNR_dB"]:.2f},{m["SSIM"]:.4f},{m["Noise_HU"]:.2f},'
                        f'{m["Bias_HU"]:.2f},{m["Contrast_HU"]:.2f},{m["CNR"]:.2f},{res.seconds:.3f}\n')

        QMessageBox.information(self, "Thành công", f"Đã xuất dữ liệu CSV thành công:\n{fn}")


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

    # 1. Kích hoạt chính sách làm tròn tỉ lệ HiDPI PassThrough
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    # 2. Khởi tạo đối tượng ứng dụng QApplication
    app = QApplication(sys.argv)

    # 3. Kích hoạt giao diện qdarktheme với màu nhấn Neon Cyan (#06b6d4)
    qdarktheme.setup_theme(custom_colors={"primary": ACCENT_CYAN})

    # 4. Hiển thị cửa sổ làm việc chính
    window = CTSimulatorApp()
    window.show()

    # 5. Vòng lặp sự kiện chính Qt
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
