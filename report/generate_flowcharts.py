"""
generate_flowcharts.py - Module vẽ sơ đồ khối thuật toán (Algorithmic Flowcharts) & Đồ thị phân tích toán học chuyên sâu
cho từng phương pháp tái tạo CT (BP, FBP, SART, SIRT, MBIR PWLS-TV, ASIR, DLIR) và phân tích Ảnh CT Thực Tế.
Phiên bản nâng cấp: Tối ưu tỷ lệ khung hình chuẩn trang A4, cỡ chữ lớn rõ nét và ký hiệu toán học chuẩn LaTeX (STIX).
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Bật bộ font toán học chuẩn LaTeX (STIX / Times)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)


# --------------------------------------------------------------------------------------
# CÔNG CỤ VẼ HỘP FLOWCHART VÀ MŨI TÊN ĐỘ TƯƠNG PHẢN CAO, CỠ CHỮ LỚN RÕ NÉT
# --------------------------------------------------------------------------------------
def _draw_box(ax, x, y, w, h, title, body="", bg="#f8fafc", border="#002060",
              title_color="#002060", body_color="#0f172a", title_size=9.4, body_size=8.5, lw=1.6, radius=0.018):
    """Vẽ một khối hộp bo góc trong toạ độ chuẩn hoá [0, 1] x [0, 1] kèm thanh tiêu đề tách biệt rõ ràng."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=bg, edgecolor=border, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    if body:
        n_title = title.count("\n") + 1
        title_frac = 0.26 if n_title == 1 else 0.34
        div_y = (y + h / 2) - h * title_frac
        # Đường kẻ mảnh phân tách tiêu đề hộp và nội dung toán học
        ax.plot([x - w * 0.46, x + w * 0.46], [div_y, div_y], color=border, lw=0.7, alpha=0.35, zorder=4)
        title_y = (y + h / 2 + div_y) / 2.0
        body_y = (div_y + (y - h / 2)) / 2.0
        ax.text(x, title_y, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color, zorder=5, linespacing=1.18)
        ax.text(x, body_y, body, ha="center", va="center", fontsize=body_size,
                color=body_color, zorder=5, linespacing=1.28)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color, zorder=5, linespacing=1.22)


def _draw_arrow(ax, x1, y1, x2, y2, label="", color="#1e293b", lw=1.8, rad=0.0, label_offset=(0.0, 0.035), fontsize=8.4):
    """Vẽ mũi tên kết nối hai khối kèm nhãn toán học đặt tách biệt khỏi đường viền hộp."""
    style = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=style,
        arrowstyle="-|>,head_length=6.5,head_width=4.2",
        color=color, linewidth=lw, zorder=6
    )
    ax.add_patch(arr)
    if label:
        mx = (x1 + x2) / 2.0 + label_offset[0]
        my = (y1 + y2) / 2.0 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor=color, linewidth=0.8, alpha=0.98))


def _setup_canvas(figsize=(8.4, 4.8), dpi=240, title=""):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0.5, 0.968, title, ha="center", va="center", fontsize=10.6,
                fontweight="bold", color="#002060")
    return fig, ax


# --------------------------------------------------------------------------------------
# 0. FLOWCHART KHUNG TOÁN HỌC BÀI TOÁN NGƯỢC TỔNG QUÁT (INVERSE PROBLEM FRAMEWORK)
# --------------------------------------------------------------------------------------
def draw_fc0_inverse_problem(out_path: str) -> str:
    fig, ax = _setup_canvas(
        figsize=(8.4, 5.0),
        title="SƠ ĐỒ KHUNG TOÁN HỌC BÀI TOÁN NGƯỢC: TỪ SUY GIẢM VẬT LÝ ĐẾN KHÔI PHỤC HÀM $\\mu(x, y)$"
    )

    # Hàng trên: Quá trình thuận (Forward Physics & Radon)
    _draw_box(ax, 0.145, 0.72, 0.25, 0.26,
              "1. ĐỐI TƯỢNG VẬT LÝ",
              "Hàm suy giảm chưa biết:\n$\\mu(x, y, E) \\in L^2(\\Omega)$\nChuẩn hoá: $\\rho = 1 + \\frac{\\mathrm{HU}}{1000}$",
              bg="#eff6ff", border="#1d4ed8", title_color="#1e40af")

    _draw_box(ax, 0.475, 0.72, 0.27, 0.26,
              "2. TOÁN TỬ THUẬN RADON ($\\mathcal{R}$)",
              "Tích phân đường tia $L(s,\\theta)$:\n$p = \\iint_{\\Omega} \\mu(x,y)\\,\\delta(x\\cos\\theta+y\\sin\\theta-s)\\,dxdy$\nTrị kỳ dị suy giảm: $\\sigma_k \\sim |\\omega|^{-1/2}$",
              bg="#f0fdf4", border="#15803d", title_color="#166534")

    _draw_box(ax, 0.825, 0.72, 0.29, 0.26,
              "3. THỐNG KÊ PHOTON & LOGARIT",
              "$N_i \\sim \\mathrm{Poisson}(I_0 e^{-p_i}) + \\mathcal{N}(0,\\sigma_e^2)$\n$p_i = -\\ln(N_i/I_0) \\Rightarrow \\mathrm{Var}(p_i) \\approx \\frac{\\bar{N}_i+\\sigma_e^2}{\\bar{N}_i^2}$\nLiều thấp ($I_0 \\downarrow$) $\\Rightarrow \\mathrm{Var}(p_i) \\uparrow\\uparrow$",
              bg="#fff7ed", border="#c2410c", title_color="#9a3412")

    _draw_arrow(ax, 0.272, 0.72, 0.338, 0.72, "Tia X", label_offset=(0.0, 0.048), fontsize=8.0)
    _draw_arrow(ax, 0.612, 0.72, 0.678, 0.72, "Beer-Lambert", label_offset=(0.0, 0.048), fontsize=8.0)

    # Mũi tên chuyển xuống bài toán ngược
    _draw_arrow(ax, 0.825, 0.585, 0.50, 0.455,
                "Bài toán Ngược đặt không chỉnh (Ill-posed: $\\mathcal{R}^{-1}$ không liên tục)",
                color="#b91c1c", label_offset=(0.04, 0.025), fontsize=8.5)

    # Hàng dưới: 3 Hệ quy chiếu giải bài toán ngược
    _draw_box(ax, 0.165, 0.21, 0.30, 0.34,
              "NHÁNH 1: NGHỊCH ĐẢO GIẢI TÍCH\n(Analytic: BP / FBP)",
              "• Nghiệm đóng:\n  $\\hat{\\mu} = \\mathcal{R}^*\\!\\left[\\mathcal{F}^{-1}\\{|\\omega|\\,W(\\omega)\\,\\mathcal{F}p\\}\\right]$\n• Ưu: Cực nhanh $\\mathcal{O}(N^3)$\n• Nhược: Nhân $|\\omega|$ khuếch đại nhiễu:\n  $\\sigma_{\\mu}^2 \\propto \\frac{1}{I_0}\\int_0^{\\omega_N} \\omega^2 W^2(\\omega)\\,d\\omega$",
              bg="#fef2f2", border="#dc2626", title_color="#991b1b", title_size=8.8, body_size=8.1)

    _draw_box(ax, 0.50, 0.21, 0.32, 0.34,
              "NHÁNH 2: TỐI ƯU ĐẠI SỐ & THỐNG KÊ\n(Iterative: SART / SIRT / MBIR)",
              "• Cực tiểu hoá hàm mục tiêu MAP:\n  $\\min_{\\mu\\geq 0} \\frac{1}{2}\\|A\\mu - p\\|_W^2 + \\beta\\,\\mathrm{TV}_{\\delta}(\\mu)$\n• Trọng số $w_i = 1/\\mathrm{Var}(p_i) \\approx \\bar{N}_i$\n• TV bảo toàn biên, khử sọc kim loại\n• Nhược: Chi phí lặp CG lớn, bậc thang",
              bg="#ecfdf5", border="#059669", title_color="#065f46", title_size=8.8, body_size=8.1)

    _draw_box(ax, 0.835, 0.21, 0.30, 0.34,
              "NHÁNH 3: HỌC SÂU PHẦN DƯ\n(Deep Learning: DLIR CNN)",
              "• Ánh xạ phần dư phi tuyến:\n  $\\hat{\\mu}_{\\mathrm{DL}} = \\hat{\\mu}_{\\mathrm{FBP}} - \\alpha\\,\\mathcal{F}_{\\Theta}(\\hat{\\mu}_{\\mathrm{FBP}})$\n• Học trường nhiễu $\\mathcal{R}_{\\mathrm{FBP}}^{-1}(\\eta_{\\mathrm{Poisson}})$\n• Ưu: Tốc độ $\\sim 0.08\\,\\mathrm{s}$, giữ trọn biên\n  & kết cấu hạt tự nhiên (NPS)",
              bg="#f0f9ff", border="#0284c7", title_color="#075985", title_size=8.8, body_size=8.1)

    _draw_arrow(ax, 0.44, 0.445, 0.165, 0.385, "", color="#dc2626")
    _draw_arrow(ax, 0.50, 0.445, 0.50, 0.385, "", color="#059669")
    _draw_arrow(ax, 0.56, 0.445, 0.835, 0.385, "", color="#0284c7")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# 1. FLOWCHART PHƯƠNG PHÁP CHIẾU NGƯỢC ĐƠN THUẦN (UNFILTERED BP)
# --------------------------------------------------------------------------------------
def draw_fc1_bp(out_path: str) -> str:
    fig, ax = _setup_canvas(
        figsize=(8.4, 3.4),
        title="FLOWCHART THUẬT TOÁN 1: CHIẾU NGƯỢC ĐƠN THUẦN (UNFILTERED BACKPROJECTION - BP)"
    )

    _draw_box(ax, 0.115, 0.47, 0.20, 0.62,
              "ĐẦU VÀO (INPUT)",
              "• Sinogram $p(s, \\theta_m)$\n  cỡ $N_{\\mathrm{det}} \\times N_{\\theta}$\n• Góc chiếu:\n  $\\theta_m \\in [0, \\pi)$",
              bg="#eff6ff", border="#1d4ed8", title_size=9.2, body_size=8.6)

    _draw_box(ax, 0.365, 0.47, 0.23, 0.62,
              "BƯỚC 1: ÁNH XẠ HÌNH HỌC",
              "Tại pixel $(x_j, y_j)$ và góc $\\theta_m$:\n$s_{jm} = x_j\\cos\\theta_m + y_j\\sin\\theta_m$\nNội suy tuyến tính 1D trên $s$:\n$\\tilde{p}(s_{jm}) = (1-w)p_{i_0} + w p_{i_0+1}$",
              bg="#f8fafc", border="#334155", title_size=9.2, body_size=8.5)

    _draw_box(ax, 0.635, 0.47, 0.23, 0.62,
              "BƯỚC 2: TÍCH PHÂN GÓC ($\\mathcal{R}^*$)",
              "Cộng dồn đồng nhất mọi góc:\n$b(x,y) = \\frac{\\Delta\\theta}{\\pi} \\sum_{m=1}^{N_\\theta} \\tilde{p}(s_{jm}, \\theta_m)$\nChuẩn hoá bảo toàn tổng:\n$\\hat{\\mu}_{\\mathrm{BP}} = b \\cdot \\left(\\sum \\mu_{\\mathrm{ref}} / \\sum b\\right)$",
              bg="#f0fdf4", border="#15803d", title_size=9.2, body_size=8.5)

    _draw_box(ax, 0.885, 0.47, 0.21, 0.62,
              "HỆ QUẢ TOÁN HỌC",
              "$b(x,y) = \\mu(x,y) * \\frac{1}{\\pi r}$\n• Phổ 2D:\n  $B(\\omega_x,\\omega_y) = \\frac{M(\\omega_x,\\omega_y)}{\\pi|\\omega|}$\n• Mờ sương $1/r$, mất biên!",
              bg="#fef2f2", border="#dc2626", title_color="#991b1b", title_size=9.2, body_size=8.5)

    _draw_arrow(ax, 0.217, 0.47, 0.248, 0.47, "$p(s,\\theta)$", label_offset=(0.0, 0.08))
    _draw_arrow(ax, 0.482, 0.47, 0.518, 0.47, "$\\tilde{p}(s_{jm})$", label_offset=(0.0, 0.08))
    _draw_arrow(ax, 0.752, 0.47, 0.778, 0.47, "PSF $1/r$", label_offset=(0.0, 0.08))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# 2. FLOWCHART PHƯƠNG PHÁP CHIẾU NGƯỢC CÓ LỌC (FBP & 5 CỬA SỔ LỌC) - BỐ CỤC 2 HÀNG RÕ NÉT
# --------------------------------------------------------------------------------------
def draw_fc2_fbp(out_path: str) -> str:
    fig, ax = _setup_canvas(
        figsize=(8.4, 5.2),
        title="FLOWCHART THUẬT TOÁN 2: CHIẾU NGƯỢC CÓ LỌC (FBP) & CƠ CHẾ 5 BỘ LỌC MIỀN TẦN SỐ"
    )

    # Hàng 1: Bước 1 -> Bước 2 -> Bước 3 (3 hộp lớn, rộng rãi)
    _draw_box(ax, 0.16, 0.72, 0.28, 0.31,
              "BƯỚC 1: ĐỆM SỐ KHÔNG\n(Zero-Padding)",
              "$M = 2^{\\lceil \\log_2(2 N_{\\mathrm{det}}) \\rceil}$\nMở rộng $p_{\\mathrm{pad}}$ cỡ $M \\times N_\\theta$\nKhử chồng phổ tích chập vòng\n(Tránh lỗi lõm đĩa Cupping)",
              bg="#eff6ff", border="#1d4ed8", title_size=9.0, body_size=8.4)

    _draw_box(ax, 0.50, 0.72, 0.29, 0.31,
              "BƯỚC 2: BIẾN ĐỔI 1D FFT\nTHEO TRỤC ĐẦU DÒ $s$",
              "$P(\\omega, \\theta) = \\mathcal{F}_{1\\mathrm{D}}\\{p_{\\mathrm{pad}}(s, \\theta)\\}$\nĐịnh lý Lát cắt Trung tâm:\n$P(\\omega,\\theta) = M(\\omega\\cos\\theta, \\omega\\sin\\theta)$",
              bg="#f8fafc", border="#334155", title_size=9.0, body_size=8.4)

    _draw_box(ax, 0.84, 0.72, 0.28, 0.31,
              "BƯỚC 3: NHÂN HÀM TRUYỀN LỌC\n$H(\\omega) = |\\omega|_{\\mathrm{Ram-Lak}} \\cdot W(\\omega)$",
              "• Nhân $|\\omega|$ bù Jacobian toạ độ cực\n• Nhân đáp ứng xung $h[0]=1/4$,\n  $h[n\\,\\mathrm{odd}] = -1/(\\pi n)^2$\n• Nhân cửa sổ điều biến $W(\\omega)$",
              bg="#fefce8", border="#ca8a04", title_color="#854d0e", title_size=9.0, body_size=8.4)

    _draw_arrow(ax, 0.302, 0.72, 0.352, 0.72, "$p_{\\mathrm{pad}}$", label_offset=(0.0, 0.045))
    _draw_arrow(ax, 0.648, 0.72, 0.698, 0.72, "$P(\\omega,\\theta)$", label_offset=(0.0, 0.045))

    # Hàng 2: Bước 4 & 5 + Khung 5 Cửa sổ điều biến W(ω)
    _draw_box(ax, 0.21, 0.24, 0.38, 0.35,
              "BƯỚC 4 & 5: BIẾN ĐỔI NGƯỢC IFFT\n& CHIẾU NGƯỢC CHUẨN HOÁ",
              "1) Lọc miền không gian:\n"
              "   $q(s,\\theta) = \\mathrm{Re}\\!\\left\\{\\mathcal{F}_{1\\mathrm{D}}^{-1}\\{P\\cdot H\\}\\right\\}[:N_{\\mathrm{det}}]$\n"
              "2) Tích phân chiếu ngược & đổi sang HU:\n"
              "   $\\hat{\\mu}_{\\mathrm{FBP}} = \\frac{\\Delta\\theta}{2}\\sum_{m=1}^{N_\\theta} q(s_{jm}, \\theta_m),\\; \\mathrm{HU}=1000(\\hat{\\mu}-1)$",
              bg="#f0fdf4", border="#15803d", title_color="#166534", title_size=8.5, body_size=7.8)

    _draw_box(ax, 0.705, 0.24, 0.55, 0.35,
              "5 CỬA SỔ ĐIỀU BIẾN $W(\\omega)$ & TRUYỀN PHƯƠNG SAI NHIỄU",
              "• Phương sai nhiễu: $\\sigma_{\\mu}^2 \\propto \\frac{1}{I_0 N_\\theta \\Delta s^3}\\int_0^{\\omega_N} \\omega^2 W^2(\\omega)\\,d\\omega$ (Giảm $\\Delta s$ đi $2\\times \\Rightarrow$ Nhiễu $\\times 8$)\n"
              "• Ramp ($W=1,\\, \\eta_W=1.00$): Sắc nét nhất  |  • Shepp-Logan ($\\mathrm{sinc},\\, \\eta_W=0.69$)\n"
              "• Cosine ($\\cos(\\frac{\\pi u}{2}),\\, \\eta_W=0.39$)  |  • Hamming ($0.54+0.46\\cos(\\pi u),\\, \\eta_W=0.23$)\n"
              "• Hann ($0.50+0.50\\cos(\\pi u),\\, \\eta_W=0.195$): Giảm $>80\\%$ phương sai nhiễu hạt",
              bg="#fff1f2", border="#e11d48", title_color="#9f1239", title_size=8.5, body_size=7.7)

    # Mũi tên nối từ Bước 3 xuống Bước 4 & Khung 5 cửa sổ
    _draw_arrow(ax, 0.84, 0.56, 0.28, 0.415, "Qua bộ lọc $H(\\omega)$", color="#15803d", rad=-0.08, label_offset=(-0.05, 0.03))
    _draw_arrow(ax, 0.84, 0.56, 0.78, 0.415, "Chọn $W(\\omega)$", color="#e11d48", label_offset=(0.06, 0.0))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# 3. FLOWCHART THUẬT TOÁN LẶP ĐẠI SỐ: SART (OS-SART) & SIRT
# --------------------------------------------------------------------------------------
def draw_fc3_sart_sirt(out_path: str) -> str:
    fig, ax = _setup_canvas(
        figsize=(8.4, 5.0),
        title="FLOWCHART THUẬT TOÁN 3 & 4: TÁI TẠO LẶP ĐẠI SỐ SART (OS-SART) VÀ SIRT"
    )

    _draw_box(ax, 0.145, 0.49, 0.25, 0.74,
              "1. MA TRẬN HỆ THỐNG $A$\n(Dải Hình Thang)",
              "• Dấu chân pixel vuông chiếu\n  theo góc $\\theta$ là hình thang:\n  $L = \\frac{|\\cos\\theta|+|\\sin\\theta|}{2}$\n  $M = \\frac{||\\cos\\theta|-|\\sin\\theta||}{2}$\n• $A_{ij} = \\mathrm{CDF}(s_i+\\frac{1}{2})-\\mathrm{CDF}(s_i-\\frac{1}{2})$\n• Tiền điều kiện đường chéo:\n  $R_{ii} = 1/\\sum_j A_{ij}$\n  $C_{jj} = 1/\\sum_i A_{ij}$",
              bg="#eff6ff", border="#1d4ed8", title_size=8.8, body_size=8.2)

    # Nhánh trên: OS-SART
    _draw_box(ax, 0.515, 0.70, 0.39, 0.34,
              "NHÁNH A: (OS-)SART (Kaczmarz Khối Góc)",
              "• Hoán vị $M$ khối góc theo Tỷ lệ vàng ($\\approx 111.2^\\circ$):\n  $\\mathrm{step} = \\mathrm{round}(0.618\\cdot M),\\; \\gcd(\\mathrm{step}, M)=1$\n• Cập nhật tức thì sau mỗi khối góc $s$:\n  $\\mu^{(k+1)} = \\mathcal{P}_{\\geq 0}\\!\\left[\\mu^{(k)} + \\lambda\\, C_s A_s^T R_s (p_s - A_s\\mu^{(k)})\\right]$",
              bg="#f0fdf4", border="#15803d", title_color="#166534", title_size=8.8, body_size=8.2)

    # Nhánh dưới: SIRT
    _draw_box(ax, 0.515, 0.26, 0.39, 0.34,
              "NHÁNH B: SIRT (Preconditioned Landweber)",
              "• Chiếu xuôi đồng thời toàn bộ $M$ tia: $\\hat{p}^{(k)} = A\\mu^{(k)}$\n• Cập nhật trung bình toàn cục:\n  $\\mu^{(k+1)} = \\mathcal{P}_{\\geq 0}\\!\\left[\\mu^{(k)} + \\lambda\\, C A^T R (p - A\\mu^{(k)})\\right]$\n• Trị riêng $\\rho(C A^T R A) \\leq 1 \\Rightarrow$ Hội tụ với $0 < \\lambda < 2$",
              bg="#faf5ff", border="#7e22ce", title_color="#6b21a8", title_size=8.8, body_size=8.2)

    _draw_box(ax, 0.86, 0.49, 0.24, 0.74,
              "BÁN HỘI TỤ PHỔ\n(Semi-Convergence)",
              "• Phân rã kỳ dị (SVD):\n  Bộ lọc phổ ở vòng lặp $k$:\n  $f_k(\\sigma_m) = 1 - (1 - \\lambda\\sigma_m^2)^k$\n• Khi $k$ nhỏ: $f_k \\approx k\\lambda\\sigma_m^2$\n  $\\Rightarrow$ Khôi phục cấu trúc,\n     chặn nhiễu tần số cao\n• Dừng sớm (Early Stopping)\n  là điều chuẩn Tikhonov ẩn!",
              bg="#fff7ed", border="#c2410c", title_color="#9a3412", title_size=8.8, body_size=8.2)

    _draw_arrow(ax, 0.272, 0.64, 0.318, 0.70, "$A_s$", color="#15803d", label_offset=(-0.01, 0.045))
    _draw_arrow(ax, 0.272, 0.34, 0.318, 0.26, "$A$", color="#7e22ce", label_offset=(-0.01, -0.045))
    _draw_arrow(ax, 0.712, 0.70, 0.738, 0.64, "1-2 vòng", color="#15803d", label_offset=(0.0, 0.05))
    _draw_arrow(ax, 0.712, 0.26, 0.738, 0.34, "30-40 vòng", color="#7e22ce", label_offset=(0.0, -0.05))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# 4. FLOWCHART THUẬT TOÁN TÁI TẠO LẶP THỐNG KÊ MBIR (PWLS-TV)
# --------------------------------------------------------------------------------------
def draw_fc4_mbir(out_path: str) -> str:
    fig, ax = _setup_canvas(
        figsize=(8.4, 5.2),
        title="FLOWCHART THUẬT TOÁN 5: TÁI TẠO LẶP THỐNG KÊ MBIR (PWLS-TV VIA LAGGED DIFFUSIVITY + PCG)"
    )

    _draw_box(ax, 0.145, 0.70, 0.25, 0.36,
              "1. MÔ HÌNH MAP\n& KHỞI TẠO ẤM",
              "• Trọng số thống kê:\n  $w_i = \\frac{1}{\\mathrm{Var}(p_i)} = \\frac{N_i^2}{N_i + \\sigma_e^2}$\n• Vế phải: $b = A^T W p$\n• Warm start: $\\mu^{(0)} = \\mu_{\\mathrm{FBP}}$",
              bg="#eff6ff", border="#1d4ed8", title_size=8.8, body_size=8.3)

    _draw_box(ax, 0.50, 0.70, 0.36, 0.36,
              "2. VÒNG LẶP NGOÀI ($k = 1..N_{\\mathrm{outer}}$):\nKHUẾCH TÁN TRỄ DỊ HƯỚNG",
              "• Sai phân hữu hạn: $D_x \\mu^{(k)},\\; D_y \\mu^{(k)}$\n• Hệ số dẫn dị hướng ($\\delta = 0.01 \\approx 10\\,\\mathrm{HU}$):\n  $c_x = \\frac{1}{\\sqrt{(D_x\\mu^{(k)})^2 + \\delta^2}},\\; c_y = \\frac{1}{\\sqrt{(D_y\\mu^{(k)})^2 + \\delta^2}}$\n• Toán tử: $L(\\mu^{(k)}) = D_x^T C_x D_x + D_y^T C_y D_y$",
              bg="#fefce8", border="#ca8a04", title_color="#854d0e", title_size=8.8, body_size=8.2)

    _draw_box(ax, 0.845, 0.70, 0.25, 0.36,
              "3. VÒNG LẶP TRONG ($N_{\\mathrm{cg}}=12$):\nGRADIENT LIÊN HỢP (CG)",
              "• Giải hệ SPD tuyến tính hoá:\n  $[A^T W A + \\beta L(\\mu^{(k)})]\\mu^{(k+1)} = b$\n• Chiếu không âm: $\\mu \\leftarrow \\max(\\mu, 0)$\n• Lặp lại tới khi hội tụ",
              bg="#f0fdf4", border="#15803d", title_color="#166534", title_size=8.8, body_size=8.3)

    _draw_arrow(ax, 0.272, 0.70, 0.318, 0.70, "$\\mu^{(0)}, W$", label_offset=(0.0, 0.05))
    _draw_arrow(ax, 0.682, 0.70, 0.718, 0.70, "$L(\\mu^{(k)})$", label_offset=(0.0, 0.05))
    # Vòng lặp phản hồi từ Bước 3 về Bước 2
    _draw_arrow(ax, 0.845, 0.515, 0.50, 0.515, "Cập nhật $\\mu^{(k+1)}$ cho vòng lặp ngoài kế tiếp",
                color="#059669", rad=0.14, label_offset=(0.0, -0.052), fontsize=8.3)

    # Hộp giải thích cơ chế vật lý - toán học bên dưới
    _draw_box(ax, 0.50, 0.19, 0.96, 0.29,
              "CƠ CHẾ VẬT LÝ – TOÁN HỌC GIÚP MBIR TRIỆT TIÊU NHIỄU MÀ BẢO TOÀN BIÊN GIẢI PHẪU",
              "1) Khử sọc kim loại (Streak Suppression): Khi tia $i$ xuyên qua kim loại bị đói photon ($N_i \\to 0$) $\\Rightarrow w_i \\to 0 \\Rightarrow$ tự động loại bỏ tia lỗi khỏi $A^T W A$.\n"
              "2) Khuếch tán dị hướng bảo toàn biên (Edge-Preserving TV): Tại nền mô mềm ($|D\\mu| \\ll \\delta$) $\\Rightarrow c_x \\approx 1/\\delta$ rất lớn $\\Rightarrow$ san phẳng nhiễu hạt.\n"
              "   Tại bờ biên giải phẫu ($|D\\mu| \\gg \\delta$) $\\Rightarrow c_x \\approx 1/|D\\mu| \\to 0 \\Rightarrow$ ngắt dòng khuếch tán qua biên, giữ nguyên độ sắc nét của bờ tổn thương!",
              bg="#ecfdf5", border="#059669", title_color="#065f46", title_size=8.9, body_size=8.2)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# 5. FLOWCHART THUẬT TOÁN ASIR (HYBRID) & HỌC SÂU PHẦN DƯ DLIR (RESIDUAL CNN)
# --------------------------------------------------------------------------------------
def draw_fc5_asir_dlir(out_path: str) -> str:
    fig, ax = _setup_canvas(
        figsize=(8.4, 5.2),
        title="FLOWCHART THUẬT TOÁN 6 & 7: TÁI TẠO LAI THỐNG KÊ (ASIR) VÀ HỌC SÂU PHẦN DƯ (DLIR - RESIDUAL CNN)"
    )

    # Nhánh trên: ASIR (2 hộp đầu vào dẫn vào hộp tổ hợp lồi NPS rõ ràng, không dùng ASCII art)
    _draw_box(ax, 0.145, 0.76, 0.25, 0.27,
              "ĐẦU VÀO NHÁNH ASIR",
              "1) $\\hat{\\mu}_{\\mathrm{FBP}}$: Sắc nét, nhiễu cao\n"
              "2) $\\hat{\\mu}_{\\mathrm{MBIR}}$: Nhiễu thấp, mịn",
              bg="#fffbeb", border="#d97706", title_color="#92400e", title_size=8.5, body_size=8.0)

    _draw_box(ax, 0.66, 0.76, 0.64, 0.27,
              "THUẬT TOÁN 6: ASIR (Tổ Hợp Lồi & Điều Khiển Phổ Nhiễu NPS)",
              "• Pha trộn tuyến tính: $\\hat{\\mu}_{\\mathrm{ASIR}}(x,y;\\alpha) = (1 - \\alpha)\\,\\hat{\\mu}_{\\mathrm{FBP}} + \\alpha\\,\\hat{\\mu}_{\\mathrm{MBIR}}$  (Chuẩn $\\alpha = 0.5$)\n"
              "• Phổ công suất nhiễu: $\\mathrm{NPS}_{\\mathrm{ASIR}}(\\omega) = (1-\\alpha)^2\\mathrm{NPS}_{\\mathrm{FBP}} + \\alpha^2\\mathrm{NPS}_{\\mathrm{MBIR}} + 2\\alpha(1-\\alpha)\\mathrm{CrossNPS}$\n"
              "• Ý nghĩa: Giảm $75\\%$ công suất nhiễu tần số cao, giữ kết cấu hạt tự nhiên",
              bg="#fffbeb", border="#d97706", title_color="#92400e", title_size=8.5, body_size=7.7)

    _draw_arrow(ax, 0.272, 0.76, 0.338, 0.76, "$\\alpha$", color="#d97706", label_offset=(0.0, 0.045))

    # Nhánh dưới: DLIR Pipeline
    _draw_box(ax, 0.115, 0.25, 0.19, 0.36,
              "1. CHUẨN HOÁ\nĐẦU VÀO",
              "Ảnh $x = \\hat{\\mu}_{\\mathrm{FBP}}$\n$\\tilde{x} = \\frac{\\mathrm{clip}(x,-1000,2000)}{200}$\nĐưa về $[-5, +10]$",
              bg="#eff6ff", border="#1d4ed8", title_size=8.6, body_size=8.1)

    _draw_box(ax, 0.385, 0.25, 0.26, 0.36,
              "2. TRÍCH ĐẶC TRƯNG\n(5 Lớp Conv $3\\times 3$ + ReLU)",
              "$h^{(l)} = \\mathrm{ReLU}(W_{l-1}*h^{(l-1)}+b_{l-1})$\nThực thi: $\\mathrm{im2col} + \\mathrm{GEMM}$\nVùng cảm thụ: $13\\times 13\\,\\mathrm{px}$",
              bg="#f0f9ff", border="#0284c7", title_color="#075985", title_size=8.6, body_size=8.1)

    _draw_box(ax, 0.665, 0.25, 0.21, 0.36,
              "3. TRƯỜNG NHIỄU\n& CỔNG BIÊN",
              "$\\hat{\\nu} = \\mathcal{F}_{\\Theta}(\\tilde{x})\\cdot e^{-(|\\nabla x|/g_0)^2}$\nHọc nhiễu chiếu ngược:\n$\\hat{\\nu} \\approx \\mathcal{R}_{\\mathrm{FBP}}^{-1}(\\eta_{\\mathrm{Poisson}})$",
              bg="#fdf4ff", border="#a21caf", title_color="#86198f", title_size=8.6, body_size=8.1)

    _draw_box(ax, 0.895, 0.25, 0.17, 0.36,
              "4. KHÔI PHỤC\n(Residual Out)",
              "$\\hat{\\mu}_{\\mathrm{DLIR}} = x - \\alpha_{\\mathrm{DL}}\\hat{\\nu}$\n• Tốc độ $\\sim 0.08\\,\\mathrm{s}$\n• Bảo toàn biên",
              bg="#f0fdf4", border="#15803d", title_color="#166534", title_size=8.6, body_size=8.1)

    _draw_arrow(ax, 0.212, 0.25, 0.252, 0.25, "$\\tilde{x}$", label_offset=(0.0, 0.048))
    _draw_arrow(ax, 0.517, 0.25, 0.558, 0.25, "$h^{(5)}$", label_offset=(0.0, 0.048))
    _draw_arrow(ax, 0.772, 0.25, 0.808, 0.25, "$\\hat{\\nu}$", label_offset=(0.0, 0.048))
    # Đường skip connection từ bước 1 sang bước 4
    _draw_arrow(ax, 0.115, 0.435, 0.895, 0.435,
                "Đường truyền tắt (Identity Skip Connection: $x = \\hat{\\mu}_{\\mathrm{FBP}}$)",
                color="#0f172a", rad=-0.09, label_offset=(0.0, 0.038), fontsize=8.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# 6. ĐỒ THỊ PHÂN TÍCH CHUYÊN SÂU 5 BỘ LỌC FBP (MIỀN TẦN SỐ, MIỀN KHÔNG GIAN & ĐÁNH ĐỔI)
# --------------------------------------------------------------------------------------
def draw_fig5_filter_analysis(out_path: str) -> str:
    from ct_sim.recon import FILTERS, fourier_filter, fbp
    from ct_sim.physics import ScanParams, simulate_scan
    from ct_sim.metrics import evaluate

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=180)

    size = 128
    freqs = np.fft.rfftfreq(size) * 2.0  # Chuẩn hoá ω / ω_Nyquist ∈ [0, 1]
    colors = {
        "ramp": "#ef4444",
        "shepp-logan": "#f59e0b",
        "cosine": "#10b981",
        "hamming": "#3b82f6",
        "hann": "#8b5cf6",
    }
    labels = {
        "ramp": "Ramp (Ram-Lak)",
        "shepp-logan": "Shepp-Logan",
        "cosine": "Cosine",
        "hamming": "Hamming",
        "hann": "Hann (Hanning)",
    }

    # (a) Đáp ứng biên độ miền tần số H(ω)
    for name in FILTERS:
        h = fourier_filter(size, name).ravel()
        h_half = h[: len(freqs)]
        ax1.plot(freqs, h_half, lw=1.8, color=colors[name], label=labels[name])
    ax1.set_title("(a) Đáp ứng Tần số $H(\\omega) = |\\omega|\\cdot W(\\omega)$", fontsize=8.8, fontweight="bold")
    ax1.set_xlabel("Tần số chuẩn hoá $\\omega / \\omega_{\\mathrm{Nyquist}}$", fontsize=8.2)
    ax1.set_ylabel("Hệ số khuếch đại biên độ $H(\\omega)$", fontsize=8.2)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=7.0, loc="upper left")
    ax1.tick_params(labelsize=7.5)

    # (b) Đáp ứng xung miền không gian h[n]
    lags = np.arange(-12, 13)
    for name in ("ramp", "shepp-logan", "hann"):
        h = fourier_filter(size, name).ravel()
        imp = np.real(np.fft.ifft(h))
        imp_shift = np.fft.fftshift(imp)[size // 2 - 12: size // 2 + 13]
        ax2.plot(lags, imp_shift, marker="o", markersize=3.5, lw=1.5, color=colors[name], label=labels[name])
    ax2.axhline(0, color="#475569", lw=0.8, linestyle="--")
    ax2.set_title("(b) Đáp ứng Xung Không gian $h[n]$", fontsize=8.8, fontweight="bold")
    ax2.set_xlabel("Độ lệch mẫu đầu dò $n$ (bins)", fontsize=8.2)
    ax2.set_ylabel("Biên độ nhân chập $h[n]$", fontsize=8.2)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=7.0)
    ax2.tick_params(labelsize=7.5)

    # (c) Đánh đổi thực nghiệm giữa Nhiễu ROI (HU) và PSNR/CNR của 5 bộ lọc ở liều thấp (60 mA)
    sc = simulate_scan("Đầu (HU)", 128, ScanParams(ma=60.0, poisson=True))
    noises, psnrs = [], []
    for name in FILTERS:
        hu_rec = 1000.0 * (fbp(sc.sino, sc.theta, name) - 1.0)
        m = evaluate(hu_rec, sc.hu_true, sc.info)
        noises.append(m["Noise_HU"])
        psnrs.append(m["PSNR_dB"])
        ax3.scatter(m["Noise_HU"], m["PSNR_dB"], color=colors[name], s=55, zorder=4, label=labels[name])
        ax3.annotate(name.capitalize(), (m["Noise_HU"], m["PSNR_dB"]),
                     textcoords="offset points", xytext=(4, 2), fontsize=7.2, fontweight="bold", color=colors[name])

    ax3.plot(noises, psnrs, color="#64748b", linestyle="--", lw=1.1, zorder=2)
    ax3.set_title("(c) Đánh đổi Nhiễu vs PSNR (Tại 60 mA)", fontsize=8.8, fontweight="bold")
    ax3.set_xlabel("Độ lệch chuẩn Nhiễu ROI $\\sigma_{\\mathrm{ROI}}$ (HU)", fontsize=8.2)
    ax3.set_ylabel("Chất lượng ảnh PSNR (dB)", fontsize=8.2)
    ax3.set_xlim(min(noises) - 1.0, max(noises) + 2.8)
    ax3.set_ylim(min(psnrs) - 0.8, max(psnrs) + 0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.tick_params(labelsize=7.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# 7. HÌNH THỰC NGHIỆM PHÂN TÍCH & TÁI TẠO TRÊN ẢNH CT LÂM SÀNG THỰC TẾ (REAL CLINICAL CT)
# --------------------------------------------------------------------------------------
def draw_fig6_real_ct_analysis(out_path: str) -> str:
    from ct_sim.physics import ScanParams, simulate_scan
    from ct_sim.pipeline import run_algorithms

    sc = simulate_scan("CT Thực Tế: Gan Bình Thường", 128, ScanParams(ma=70.0, poisson=True))
    res = run_algorithms(sc, ["FBP", "SART", "MBIR", "DLIR"])

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), dpi=180)
    lvl, wid = sc.info.window
    vmin, vmax = lvl - wid / 2, lvl + wid / 2

    # 1. Ảnh CT Thực tế gốc + Vòng tròn ROI
    ax = axes[0, 0]
    ax.imshow(sc.hu_true, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    for rname, rcol, rlbl in [("bg", "#38bdf8", "ROI Nền (Gan)"), ("lesion", "#facc15", "ROI Tương phản")]:
        cx, cy, r = sc.info.rois[rname]
        px = (cx + 1.0) * 0.5 * sc.n
        py = (1.0 - cy) * 0.5 * sc.n
        pr = r * 0.5 * sc.n
        circ = plt.Circle((px, py), pr, fill=False, edgecolor=rcol, linewidth=1.8, label=rlbl)
        ax.add_patch(circ)
    ax.set_title("1. Ảnh CT Thực Tế Gốc (Gan/Bụng)\nĐịnh vị tự động ROI Nền & Tương phản", fontsize=8.2, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.8)
    ax.axis("off")

    # 2. Sinogram thu nhận của ảnh CT thực tế
    ax = axes[0, 1]
    ax.imshow(sc.sino, cmap="inferno", aspect="auto", origin="lower", extent=[0, 180, -64, 64])
    ax.set_title("2. Biến đổi Radon $p(s,\\theta)$\n(Thu nhận ở liều thấp 70 mA)", fontsize=8.2, fontweight="bold")
    ax.set_xlabel("Góc quét $\\theta$ (độ)", fontsize=7.8)
    ax.set_ylabel("Đầu dò $s$", fontsize=7.8)
    ax.tick_params(labelsize=6.8)

    # 3. FBP (Ramp)
    r_fbp = res["FBP"]
    ax = axes[0, 2]
    ax.imshow(r_fbp.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"3. FBP (Ramp)\nPSNR: {r_fbp.metrics['PSNR_dB']:.1f} dB | Noise: {r_fbp.metrics['Noise_HU']:.1f} HU",
                 fontsize=8.2, fontweight="bold")
    ax.axis("off")

    # 4. SART
    r_sart = res["SART"]
    ax = axes[1, 0]
    ax.imshow(r_sart.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"4. SART (Khối góc Kaczmarz)\nPSNR: {r_sart.metrics['PSNR_dB']:.1f} dB | Noise: {r_sart.metrics['Noise_HU']:.1f} HU",
                 fontsize=8.2, fontweight="bold")
    ax.axis("off")

    # 5. MBIR (PWLS-TV)
    r_mbir = res["MBIR"]
    ax = axes[1, 1]
    ax.imshow(r_mbir.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"5. MBIR (PWLS-TV Thống kê)\nPSNR: {r_mbir.metrics['PSNR_dB']:.1f} dB | Noise: {r_mbir.metrics['Noise_HU']:.1f} HU",
                 fontsize=8.2, fontweight="bold")
    ax.axis("off")

    # 6. DLIR
    r_dl = res["DLIR"]
    ax = axes[1, 2]
    ax.imshow(r_dl.hu, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"6. DLIR (Residual CNN)\nPSNR: {r_dl.metrics['PSNR_dB']:.1f} dB | Noise: {r_dl.metrics['Noise_HU']:.1f} HU",
                 fontsize=8.2, fontweight="bold")
    ax.axis("off")

    plt.suptitle("PHÂN TÍCH & KHÔI PHỤC HÀM SUY GIẢM $\\mu(x,y)$ TRÊN ẢNH CT LÂM SÀNG THỰC TẾ", fontsize=9.8, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_flowcharts_and_deep_figs(output_dir: str) -> Dict[str, str]:
    """Tạo toàn bộ 6 Flowchart thuật toán và 2 Hình phân tích chuyên sâu."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    paths["fc0_inverse"] = draw_fc0_inverse_problem(os.path.join(output_dir, "fc0_inverse_problem_framework.png"))
    paths["fc1_bp"] = draw_fc1_bp(os.path.join(output_dir, "fc1_bp_flowchart.png"))
    paths["fc2_fbp"] = draw_fc2_fbp(os.path.join(output_dir, "fc2_fbp_flowchart.png"))
    paths["fc3_sart_sirt"] = draw_fc3_sart_sirt(os.path.join(output_dir, "fc3_sart_sirt_flowchart.png"))
    paths["fc4_mbir"] = draw_fc4_mbir(os.path.join(output_dir, "fc4_mbir_pwls_tv_flowchart.png"))
    paths["fc5_asir_dlir"] = draw_fc5_asir_dlir(os.path.join(output_dir, "fc5_asir_dlir_flowchart.png"))
    paths["fig5_filters"] = draw_fig5_filter_analysis(os.path.join(output_dir, "fig5_filter_frequency_spatial.png"))
    paths["fig6_real_ct"] = draw_fig6_real_ct_analysis(os.path.join(output_dir, "fig6_real_ct_analysis.png"))
    return paths


if __name__ == "__main__":
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    res = generate_all_flowcharts_and_deep_figs(fig_dir)
    for k, v in res.items():
        print(f"Generated {k}: {v}")
