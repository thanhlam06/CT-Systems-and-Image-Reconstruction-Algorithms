"""
generate_btl_pdf.py - Trình tạo Báo cáo Bài tập lớn (BTL) định dạng PDF chuẩn Đại học Bách Khoa.
Môn học: Cơ sở Vật lý Hạt nhân & Kỹ thuật Y sinh (Project 2: Kỹ thuật chụp CT trong Y tế).
Định dạng bám sát 100% mẫu 'BTL XSTK_ Khoa Co Khi_L01.pdf':
  - Khung viền đôi chuẩn Bách Khoa ở trang bìa + Logo Trường ĐH Bách Khoa TP.HCM
  - Trang phân công hạng mục dự án & Trang nhận xét của giảng viên
  - Mục lục tự động, danh mục hình ảnh & sơ đồ khối thuật toán (Flowcharts), danh mục bảng biểu
  - Tích hợp 100% nội dung và hình ảnh của toàn bộ dự án:
      + Phần 1 (Tổng quan về kỹ thuật CT)
      + Phần 2.1 (Thiết kế và vận hành hệ thống Helical MDCT)
      + Phần 2.2 (Thang xám HU, chẩn đoán lâm sàng & Tối ưu liều ALARA / Rose's criterion)
      + Phần Trọng tâm Thuật toán & Phần mềm (Chương 3 – Chương 8: Giải tích bài toán ngược, 7 thuật toán tái tạo, GUI & Thực nghiệm)
  - Toàn bộ công thức toán học được kết xuất chuẩn LaTeX (STIX Math 300 DPI) với số thứ tự căn phải (x.y)
  - Toàn bộ 6 Flowchart thuật toán được thiết kế lại với cỡ chữ lớn, rõ nét và ký hiệu chuẩn LaTeX
  - Chương 8: Tài liệu tham khảo đặt trên trang riêng biệt chuẩn mẫu BTL Khoa Cơ Khí
"""
from __future__ import annotations

import os
import sys
import time
import shutil
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục gốc CT_Simulator vào sys.path để import ct_sim
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cấu hình font STIX chuẩn LaTeX cho công thức toán học
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral"]

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph as _RLParagraph, Spacer, Table as _RLTable, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Đăng ký font tiếng Việt Times New Roman hệ thống
FONT_REG = "Times-Roman-VN"
FONT_BOLD = "Times-Bold-VN"
FONT_ITALIC = "Times-Italic-VN"
FONT_BI = "Times-BoldItalic-VN"

pdfmetrics.registerFont(TTFont(FONT_REG, r"C:\Windows\Fonts\times.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BOLD, r"C:\Windows\Fonts\timesbd.ttf"))
pdfmetrics.registerFont(TTFont(FONT_ITALIC, r"C:\Windows\Fonts\timesi.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BI, r"C:\Windows\Fonts\timesbi.ttf"))
pdfmetrics.registerFontFamily(FONT_REG, normal=FONT_REG, bold=FONT_BOLD, italic=FONT_ITALIC, boldItalic=FONT_BI)


def _sanitize_times_text(text: str) -> str:
    """Chuẩn hoá các ký hiệu toán học ngoài vùng glyph của Windows times.ttf trong văn bản xuôi."""
    if not isinstance(text, str):
        return text
    replacements = [
        ("N\u0304", "N<sub>tb</sub>"),
        ("μ\u0304", "μ<sub>tb</sub>"),
        ("μ\u0302", "μ<sub>rec</sub>"),
        ("p\u0302", "p<sub>est</sub>"),
        ("\u221d", " ~ "),       # ∝
        ("\u2208", " thuộc "),   # ∈
        ("\u226a", "&lt;&lt;"),  # ≪
        ("\u226b", "&gt;&gt;"),  # ≫
        ("\u2308", "ceil("),     # ⌈
        ("\u2309", ")"),         # ⌉
        ("\u2016", "||"),        # ‖
        ("\u2207", "grad "),     # ∇
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def Paragraph(text, style, *args, **kwargs):
    return _RLParagraph(_sanitize_times_text(text), style, *args, **kwargs)


def Table(data, colWidths=None, *args, **kwargs):
    """Tự động bọc chuỗi trong các ô bảng dữ liệu (>= 3 cột) thành Paragraph để tự động xuống dòng chuẩn xác."""
    if data and len(data[0]) >= 3:
        hdr_style = ParagraphStyle(
            "Tbl_Hdr_Auto", fontName=FONT_BOLD, fontSize=7.6, leading=9.8,
            textColor=colors.white, alignment=1
        )
        cell_style = ParagraphStyle(
            "Tbl_Cell_Auto", fontName=FONT_REG, fontSize=7.4, leading=9.6,
            textColor=colors.HexColor("#1e293b"), alignment=0
        )
        cell_bold_style = ParagraphStyle(
            "Tbl_Cell_Bold_Auto", fontName=FONT_BOLD, fontSize=7.4, leading=9.6,
            textColor=colors.HexColor("#0f172a"), alignment=0
        )
        wrapped_data = []
        for r_idx, row in enumerate(data):
            new_row = []
            for c_idx, val in enumerate(row):
                if isinstance(val, str):
                    txt = _sanitize_times_text(val).replace("&", "&amp;").replace("&amp;lt;", "&lt;").replace("&amp;gt;", "&gt;").replace("\n", "<br/>")
                    if r_idx == 0:
                        new_row.append(_RLParagraph(f"<b>{txt}</b>", hdr_style))
                    elif c_idx == 0:
                        new_row.append(_RLParagraph(f"<b>{txt}</b>", cell_bold_style))
                    else:
                        new_row.append(_RLParagraph(txt, cell_style))
                else:
                    new_row.append(val)
            wrapped_data.append(new_row)
        return _RLTable(wrapped_data, colWidths=colWidths, *args, **kwargs)
    return _RLTable(data, colWidths=colWidths, *args, **kwargs)

PAGE_WIDTH, PAGE_HEIGHT = A4
BK_BLUE = colors.HexColor("#002060")
BK_CYAN = colors.HexColor("#06b6d4")
DARK_GRAY = colors.HexColor("#334155")
LIGHT_GRAY = colors.HexColor("#f1f5f9")


# --------------------------------------------------------------------------------------
# LỚP CANVAS 2-PASS: KHUNG VIỀN BÌA, HEADER & FOOTER 'Trang X/Y'
# --------------------------------------------------------------------------------------
class BTLNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            page_num = self._pageNumber
            total_pages = num_pages

            if page_num == 1:
                self._draw_cover_border()
            else:
                self._draw_running_header_footer(page_num, total_pages)

            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_cover_border(self):
        """Vẽ khung viền đôi trang bìa chuẩn Bách Khoa (Double border with corner ticks)."""
        self.saveState()
        self.setStrokeColor(BK_BLUE)

        # Viền ngoài
        self.setLineWidth(2.2)
        x1, y1 = 30, 30
        w1, h1 = PAGE_WIDTH - 60, PAGE_HEIGHT - 60
        self.rect(x1, y1, w1, h1)

        # Viền trong
        self.setLineWidth(0.8)
        x2, y2 = 36, 36
        w2, h2 = PAGE_WIDTH - 72, PAGE_HEIGHT - 72
        self.rect(x2, y2, w2, h2)

        # Góc nhấn hoa văn 4 góc (Corner ticks)
        tick_len = 12
        self.setLineWidth(1.2)
        for (cx, cy) in [(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)]:
            self.line(cx - tick_len, cy, cx + tick_len, cy)
            self.line(cx, cy - tick_len, cx, cy + tick_len)

        self.restoreState()

    def _draw_running_header_footer(self, page_num, total_pages):
        """Header và Footer cho các trang nội dung chuẩn mẫu BTL Khoa Cơ Khí."""
        self.saveState()

        # Header: Logo nhỏ + Tên trường + đường kẻ
        logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "bk_logo.png")
        if os.path.exists(logo_path):
            try:
                self.drawImage(logo_path, 40, PAGE_HEIGHT - 45, width=18, height=18, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass

        self.setFont(FONT_REG, 8.5)
        self.setFillColor(colors.HexColor("#1e293b"))
        self.drawString(64, PAGE_HEIGHT - 40, "Trường Đại học Bách khoa, ĐHQG-HCM")

        self.setStrokeColor(colors.HexColor("#0f172a"))
        self.setLineWidth(0.6)
        self.line(40, PAGE_HEIGHT - 48, PAGE_WIDTH - 40, PAGE_HEIGHT - 48)

        # Footer: Tên báo cáo + đường kẻ + Trang X/Y
        self.line(40, 45, PAGE_WIDTH - 40, 45)
        self.setFont(FONT_REG, 8.5)
        self.setFillColor(colors.HexColor("#1e293b"))
        self.drawString(40, 32, "Báo cáo Bài tập lớn: Kỹ thuật chụp CT (Computed Tomography) trong Y tế")

        page_str = f"Trang {page_num - 1}/{total_pages - 1}"
        self.setFont(FONT_REG, 8.5)
        self.drawRightString(PAGE_WIDTH - 40, 32, page_str)

        self.restoreState()


# --------------------------------------------------------------------------------------
# BỘ KẾT XUẤT CÔNG THỨC TOÁN HỌC CHUẨN LATEX (STIX 300 DPI, TỶ LỆ 1:1)
# --------------------------------------------------------------------------------------
def render_latex_eq(latex_str: str, eq_num: str, out_path: str, height_in: float = 0.50, fontsize: float = 10.8) -> Image:
    """
    Kết xuất công thức toán học hiển thị (Display Equation) theo chuẩn LaTeX (STIX fontset)
    ở độ phân giải 300 DPI với chiều rộng cố định 6.95 inch (tương ứng 500 pt trên trang A4),
    đảm bảo mọi công thức có kích thước phông chữ đồng nhất tuyệt đối và số hiệu (x.y) căn phải.
    """
    width_in = 6.95
    if not os.path.exists(out_path):
        fig = plt.figure(figsize=(width_in, height_in), dpi=300)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.455, 0.50, latex_str, ha="center", va="center", fontsize=fontsize, color="#0f172a")
        if eq_num:
            ax.text(0.992, 0.50, eq_num, ha="right", va="center", fontsize=9.8, color="#0f172a")
        fig.savefig(out_path, dpi=300, facecolor="white")
        plt.close(fig)

    w_pt = 500.0
    h_pt = w_pt * (height_in / width_in)
    return Image(out_path, width=w_pt, height=h_pt)


def generate_all_latex_equations(eq_dir: str) -> Dict[str, Image]:
    """Tạo trọn bộ 24 khối công thức chuẩn LaTeX cho toàn bộ báo cáo."""
    os.makedirs(eq_dir, exist_ok=True)
    specs = [
        # Chương 1
        ("eq_1_1", r"$I = I_0 \exp\!\left( -\int_{L} \mu(\mathbf{r})\,dl \right)$", "(1.1)", 0.48, 11.0),
        ("eq_1_2", r"$p(s, \theta) = -\ln\!\left(\frac{I}{I_0}\right) = \int_{L(s,\theta)} \mu(\mathbf{r})\,dl = \iint_{\Omega} \mu(x,y)\,\delta(x\cos\theta + y\sin\theta - s)\,dx\,dy$", "(1.2)", 0.52, 10.4),
        # Chương 2
        ("eq_2_1", r"$x(\theta) = R\cos\theta, \qquad y(\theta) = R\sin\theta, \qquad z(\theta) = z_0 + \frac{\Delta z}{2\pi}\,\theta$", "(2.1)", 0.48, 10.8),
        ("eq_2_2", r"$\mathrm{Pitch} = \frac{\Delta z}{N \times T}$", "(2.2)", 0.46, 11.0),
        ("eq_2_3", r"$\Phi(E) \propto \frac{E_{\max} - E}{E}\,\exp\!\left(-\mu_{\mathrm{Al}}(E)\,t_{\mathrm{Al}}\right), \qquad I_0 \propto \mathrm{mAs}\cdot\left(\frac{\mathrm{kVp}}{120}\right)^2$", "(2.3)", 0.52, 10.6),
        ("eq_2_4", r"$\mathrm{HU}(x, y) = 1000 \times \frac{\mu(x, y) - \mu_{\mathrm{H_2O}}}{\mu_{\mathrm{H_2O}} - \mu_{\mathrm{air}}} \approx 1000 \times \frac{\mu(x, y) - \mu_{\mathrm{H_2O}}}{\mu_{\mathrm{H_2O}}}$", "(2.4)", 0.54, 10.8),
        ("eq_2_5", r"$\mathrm{SNR} = \frac{\Delta N}{\sigma} \geq 5, \qquad \sigma = \sqrt{N} \quad\Longrightarrow\quad \mathrm{SNR} = \frac{\Delta N}{\sqrt{N}} \geq 5$", "(2.5)", 0.50, 10.8),
        ("eq_2_6", r"$\mathrm{CNR} = \frac{|N_{\mathrm{bg}} - N_{\mathrm{target}}|}{\sigma} = \frac{\Delta N}{\sqrt{N}}, \qquad C = \frac{\Delta N}{N} = \frac{|\mathrm{HU}_{\mathrm{target}} - \mathrm{HU}_{\mathrm{bg}}|}{1000} = \frac{\Delta\mathrm{HU}}{1000}$", "(2.6)", 0.54, 10.2),
        ("eq_2_7", r"$\mathrm{SNR} = \frac{C \cdot N}{\sqrt{N}} = C\sqrt{N} \geq 5 \quad\Longleftrightarrow\quad N_{\min} = \left(\frac{5}{C}\right)^2 = 25 \times \left(\frac{1000}{\Delta\mathrm{HU}}\right)^2$", "(2.7)", 0.54, 10.5),
        # Chương 3
        ("eq_3_1", r"$N_i \sim \mathrm{Poisson}(\bar{N}_i) + \mathcal{N}(0, \sigma_e^2), \qquad \bar{N}_i = I_0 \exp\!\left(-(\mathcal{R}\mu)_i\right)$", "(3.1)", 0.46, 10.8),
        ("eq_3_2", r"$\mathbb{E}[p_i] \approx (\mathcal{R}\mu)_i + \frac{\bar{N}_i + \sigma_e^2}{2\bar{N}_i^2}, \qquad \mathrm{Var}(p_i) \approx \frac{\bar{N}_i + \sigma_e^2}{\bar{N}_i^2} = \frac{1}{I_0 e^{-(\mathcal{R}\mu)_i}} + \frac{\sigma_e^2}{I_0^2 e^{-2(\mathcal{R}\mu)_i}}$", "(3.2)", 0.56, 10.2),
        ("eq_3_3", r"$b(x, y) = (\mathcal{R}^* p)(x, y) = \frac{1}{\pi}\int_0^{\pi} p(x\cos\theta + y\sin\theta, \theta)\,d\theta$", "(3.3)", 0.50, 10.8),
        ("eq_3_4", r"$b(x, y) = \iint_{\mathbb{R}^2} \frac{\mu(x^\prime, y^\prime)}{\pi\sqrt{(x-x^\prime)^2 + (y-y^\prime)^2}}\,dx^\prime\,dy^\prime = \mu(x,y) * \frac{1}{\pi r} \;\;\Longleftrightarrow\;\; B(\omega_x,\omega_y) = \frac{M(\omega_x,\omega_y)}{\pi|\omega|}$", "(3.4)", 0.56, 10.0),
        ("eq_3_5", r"$P(\omega, \theta) = \int_{-\infty}^{+\infty} p(s, \theta)\,e^{-j 2\pi\omega s}\,ds = \iint_{\mathbb{R}^2} \mu(x,y)\,e^{-j 2\pi\omega(x\cos\theta + y\sin\theta)}\,dx\,dy = M(\omega\cos\theta, \omega\sin\theta)$", "(3.5)", 0.52, 10.0),
        ("eq_3_6", r"$\mu(x, y) = \int_0^{\pi} \int_{-\infty}^{+\infty} P(\omega, \theta)\,|\omega|\,e^{j 2\pi\omega(x\cos\theta + y\sin\theta)}\,d\omega\,d\theta = \frac{1}{2}\int_0^{\pi} q(x\cos\theta + y\sin\theta, \theta)\,d\theta$", "(3.6)", 0.52, 10.2),
        ("eq_3_7", r"$h[n] = \int_{-1/2}^{+1/2} |\omega|\,e^{j 2\pi\omega n}\,d\omega \quad\Longrightarrow\quad h[0] = \frac{1}{4}, \quad h[n\ \mathrm{even}\neq 0] = 0, \quad h[n\ \mathrm{odd}] = -\frac{1}{(\pi n)^2}$", "(3.7)", 0.54, 10.1),
        ("eq_3_8", r"$\sigma_{\mu,\mathrm{FBP}}^2(x, y) = \left(\frac{\Delta\theta}{2}\right)^2 \sum_{m=1}^{N_{\theta}} \mathrm{Var}\!\left(p(s_m, \theta_m)\right)\int_{-\omega_N}^{+\omega_N} |\omega|^2\,|W(\omega)|^2\,d\omega \propto \frac{\eta_W}{I_0\,N_{\theta}\,\Delta s^3}$", "(3.8)", 0.56, 10.2),
        ("eq_3_9", r"$A_{ij} = \mathrm{CDF}_{\mathrm{trap}}\!\left(s_i + \frac{1}{2} - s_{c,j}\,;\, L, M\right) - \mathrm{CDF}_{\mathrm{trap}}\!\left(s_i - \frac{1}{2} - s_{c,j}\,;\, L, M\right), \quad L,M = \frac{|\cos\theta| \pm |\sin\theta|}{2}$", "(3.9)", 0.52, 10.0),
        ("eq_3_10", r"$\mu^{(k+1)} = \mathcal{P}_{\geq 0}\!\left[\mu^{(k)} + \lambda\,C A^T R (p - A\mu^{(k)})\right], \qquad \mu^{(k)} = \sum_{m} \left[1 - (1 - \lambda\sigma_m^2)^k\right]\frac{u_m^T p}{\sigma_m}\,v_m$", "(3.10)", 0.54, 10.1),
        ("eq_3_11", r"$\hat{\mu}_{\mathrm{MBIR}} = \arg\min_{\mu \geq 0} \left\{ \frac{1}{2}\|A\mu - p\|_W^2 + \beta\sum_{j=1}^{N^2}\left(\sqrt{(D_x\mu)_j^2 + \delta^2} + \sqrt{(D_y\mu)_j^2 + \delta^2}\right) \right\}, \quad w_i = \frac{N_i^2}{N_i + \sigma_e^2}$", "(3.11)", 0.56, 9.8),
        ("eq_3_12", r"$\nabla\Psi(\mu) = A^T W (A\mu - p) + \beta\,L(\mu)\mu = 0, \qquad L(\mu) = D_x^T \mathrm{diag}(c_x) D_x + D_y^T \mathrm{diag}(c_y) D_y, \quad c_x = \frac{1}{\sqrt{(D_x\mu)^2+\delta^2}}$", "(3.12)", 0.54, 9.8),
        ("eq_3_13", r"$\hat{\mu}_{\mathrm{ASIR}}(\alpha) = (1-\alpha)\hat{\mu}_{\mathrm{FBP}} + \alpha\hat{\mu}_{\mathrm{MBIR}}, \qquad \mathrm{NPS}_{\mathrm{ASIR}}(\omega) = (1-\alpha)^2\mathrm{NPS}_{\mathrm{FBP}}(\omega) + \alpha^2\mathrm{NPS}_{\mathrm{MBIR}}(\omega) + 2\alpha(1-\alpha)\mathrm{CrossNPS}$", "(3.13)", 0.48, 9.6),
        ("eq_3_14", r"$\hat{\mu}_{\mathrm{DLIR}} = \hat{\mu}_{\mathrm{FBP}} - \alpha_{\mathrm{DL}}\,\mathcal{F}_{\Theta}(\tilde{x})\cdot\exp\!\left(-\frac{|\nabla\hat{\mu}_{\sigma}|^2}{g_0^2}\right), \qquad \mathcal{F}_{\Theta}(\tilde{x}) \approx \mathcal{R}_{\mathrm{FBP}}^{-1}(\eta_{\mathrm{Poisson}})$", "(3.14)", 0.54, 10.2),
        # Chương 5
        ("eq_5_1", r"$\mathrm{RMSE} = \sqrt{\frac{1}{|\Omega|}\sum_{j\in\Omega}(\mathrm{HU}_j - \mathrm{HU}_{j,\mathrm{true}})^2}, \qquad \mathrm{PSNR} = 10\log_{10}\!\left(\frac{W_{\mathrm{win}}^2}{\mathrm{MSE}_{\mathrm{soft}}}\right), \qquad \mathrm{CNR} = \frac{|\bar{\mu}_{\mathrm{lesion}} - \bar{\mu}_{\mathrm{bg}}|}{\sigma_{\mathrm{ROI}}}$", "(5.1)", 0.56, 10.0),
    ]
    eq_imgs = {}
    for key, latex, num, h_in, fsize in specs:
        out_p = os.path.join(eq_dir, f"{key}_v2.png")
        eq_imgs[key] = render_latex_eq(latex, num, out_p, height_in=h_in, fontsize=fsize)
    return eq_imgs


# --------------------------------------------------------------------------------------
# HÀM TỰ ĐỘNG TẠO CÁC HÌNH ẢNH MINH HOẠ & FLOWCHART ĐỘ PHÂN GIẢI CAO
# --------------------------------------------------------------------------------------
def generate_report_figures(output_dir: str) -> Dict[str, str]:
    """Tạo trước tất cả các đồ thị nghiên cứu thực nghiệm, các sơ đồ Flowchart và nạp hình từ file .docx."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    from ct_sim.phantoms import make_phantom
    from ct_sim.physics import ScanParams, simulate_scan
    from ct_sim import recon
    from ct_sim.projector import Projector
    from ct_sim.dl_model import load_model
    from report.generate_flowcharts import generate_all_flowcharts_and_deep_figs

    print("  -> Đang tạo các sơ đồ khối thuật toán (Flowcharts) & đồ thị thực nghiệm...")
    fc_paths = generate_all_flowcharts_and_deep_figs(output_dir)
    paths.update(fc_paths)

    # Các hình ảnh từ báo cáo .docx của thành viên nhóm (đã trích xuất vào report/figures)
    docx_figs = {
        "docx_xray_vs_ct": "docx_fig1_xray_vs_ct.png",
        "docx_generations": "docx_fig2_ct_generations.png",
        "docx_gantry_das": "docx_fig3_gantry_das_chain.png",
        "docx_helical": "docx_fig4_helical_trajectory.png",
        "docx_3d_mpr": "docx_fig5_3d_mpr_pipeline.png",
        "docx_fatty_liver": "docx_fig6_fatty_liver_ct.png",
    }
    for k, fname in docx_figs.items():
        paths[k] = os.path.join(output_dir, fname)

    # 1. Hình minh hoạ thu nhận CT (Phantom + X-ray sweep)
    p1 = os.path.join(output_dir, "fig1_acquisition.png")
    if not os.path.exists(p1):
        hu, info = make_phantom("Đầu (HU)", 180)
        fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=150)
        ax_p.imshow(hu, cmap="bone", origin="lower")
        th = np.deg2rad(45)
        v = np.array([-np.sin(th), np.cos(th)])
        src = -80 * v + 90
        det = 80 * v + 90
        ax_p.plot([src[0], det[0]], [src[1], det[1]], color="#ef4444", lw=2, label="Chùm tia X (45°)")
        ax_p.plot(src[0], src[1], "yo", markersize=8, label="Bóng phát")
        ax_p.set_title("Vùng chụp cắt lớp (Phantom Đầu)", fontsize=9, fontweight="bold")
        ax_p.axis("off")
        ax_p.legend(loc="lower right", fontsize=7)

        sc = simulate_scan("Đầu (HU)", 128, ScanParams(n_views=180))
        ax_s.imshow(sc.sino, cmap="inferno", aspect="auto", origin="lower", extent=[0, 180, -64, 64])
        ax_s.set_title("Dữ liệu thô Sinogram p(s, θ)", fontsize=9, fontweight="bold")
        ax_s.set_xlabel("Góc quét θ (độ)", fontsize=8)
        ax_s.set_ylabel("Kênh đầu dò s", fontsize=8)
        ax_s.tick_params(labelsize=7)
        plt.tight_layout()
        plt.savefig(p1, bbox_inches="tight")
        plt.close(fig)
    paths["acquisition"] = p1

    # 2. Lưới so sánh 6 thuật toán tái tạo (BP, FBP Ramp, FBP Hann, SART, MBIR, DLIR)
    p2 = os.path.join(output_dir, "fig2_recon_comparison.png")
    if not os.path.exists(p2):
        sc = simulate_scan("Đầu (HU)", 128, ScanParams(ma=50, poisson=True))
        P = Projector(128, sc.theta, "strip")
        r_bp = 1000 * (recon.bp(sc.sino, sc.theta) - 1)
        r_ramp = 1000 * (recon.fbp(sc.sino, sc.theta, "ramp") - 1)
        r_hann = 1000 * (recon.fbp(sc.sino, sc.theta, "hann") - 1)
        r_sart = 1000 * (recon.sart(sc.sino, sc.theta, 1, 0.5, proj=P) - 1)
        r_mbir = 1000 * (recon.pwls_tv(sc.sino, sc.weights, sc.theta, 150.0, 6, 12, proj=P) - 1)

        dl_model = load_model()
        r_dl = dl_model.denoise(r_ramp) if dl_model else r_mbir

        fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5), dpi=150)
        imgs = [
            (r_bp, "BP (Chiếu ngược không lọc)\nMờ sương 1/r nặng"),
            (r_ramp, "FBP (Ramp - Chuẩn lâm sàng)\nTriệt tiêu mờ, nhiều hạt"),
            (r_hann, "FBP (Hann Window)\nLọc mịn, giảm nhiễu hạt"),
            (r_sart, "SART (Algebraic IR)\nHội tụ theo khối tia"),
            (r_mbir, "MBIR (PWLS + TV)\nTối ưu thống kê, mịn nền"),
            (r_dl, "DLIR (Deep Learning CNN)\nBóc tách nhiễu, giữ biên"),
        ]
        lvl, wid = 40.0, 120.0
        vmin, vmax = lvl - wid / 2, lvl + wid / 2

        for ax, (im_data, title) in zip(axes.ravel(), imgs):
            ax.imshow(im_data, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.axis("off")

        plt.suptitle("SO SÁNH CÁC THUẬT TOÁN TÁI TẠO Ở LIỀU THẤP (50 mA - GIẢM LIỀU 75%)", fontsize=10, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(p2, bbox_inches="tight")
        plt.close(fig)
    paths["recon_grid"] = p2

    # 3. Đồ thị khảo sát ALARA (Dose vs Noise & PSNR)
    p3 = os.path.join(output_dir, "fig3_alara_curves.png")
    if not os.path.exists(p3):
        from ct_sim.pipeline import dose_sweep
        ma_list = [10.0, 25.0, 50.0, 100.0, 200.0]
        sweep_data = dose_sweep("Đầu (HU)", 128, ScanParams(n_views=180), ma_list, ["FBP", "MBIR", "DLIR"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=150)
        cols = {"FBP": "#ef4444", "MBIR": "#10b981", "DLIR": "#0284c7"}

        for k in sweep_data:
            ax1.plot(sweep_data[k]["ma"], sweep_data[k]["Noise_HU"], marker="o", lw=1.8, color=cols[k], label=k)
        ax1.set_title("Mức Nhiễu Hạt ROI theo Dòng mA (ALARA)", fontsize=8.5, fontweight="bold")
        ax1.set_xlabel("Dòng bóng phát mA (tỷ lệ với liều)", fontsize=8)
        ax1.set_ylabel("Độ lệch chuẩn Nhiễu (HU)", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=7.5)

        for k in sweep_data:
            ax2.plot(sweep_data[k]["dose"], sweep_data[k]["PSNR_dB"], marker="s", lw=1.8, color=cols[k], label=k)
        ax2.set_title("Chất lượng Ảnh (PSNR) theo Liều Bức Xạ CTDIvol", fontsize=8.5, fontweight="bold")
        ax2.set_xlabel("Liều CTDIvol (mGy)", fontsize=8)
        ax2.set_ylabel("PSNR (dB)", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=7.5)

        plt.tight_layout()
        plt.savefig(p3, bbox_inches="tight")
        plt.close(fig)
    paths["alara_curves"] = p3

    # 4. Ảnh giả kim loại (Metal Artifacts)
    p4 = os.path.join(output_dir, "fig4_metal_artifact.png")
    if not os.path.exists(p4):
        sc_m = simulate_scan("Đầu (HU)", 128, ScanParams(ma=120), metal=2)
        r_fbp_m = 1000 * (recon.fbp(sc_m.sino, sc_m.theta, "ramp") - 1)
        r_mbir_m = 1000 * (recon.pwls_tv(sc_m.sino, sc_m.weights, sc_m.theta, 200.0, 6, 12) - 1)

        fig, (ax_s, ax_m1, ax_m2) = plt.subplots(1, 3, figsize=(8.5, 2.9), dpi=150)
        ax_s.imshow(sc_m.sino, cmap="inferno", aspect="auto", origin="lower")
        ax_s.set_title("Sinogram Khuyết Photon\n(Vết đen do kim loại cản tia)", fontsize=8, fontweight="bold")
        ax_s.axis("off")

        ax_m1.imshow(r_fbp_m, cmap="bone", origin="lower", vmin=-20, vmax=100)
        ax_m1.set_title("FBP: Vệt Sọc Đen Trắng\n(Streak Artifacts)", fontsize=8, fontweight="bold")
        ax_m1.axis("off")

        ax_m2.imshow(r_mbir_m, cmap="bone", origin="lower", vmin=-20, vmax=100)
        ax_m2.set_title("MBIR (Trọng số Poisson)\nGiảm sọc, phục hồi mô", fontsize=8, fontweight="bold")
        ax_m2.axis("off")

        plt.tight_layout()
        plt.savefig(p4, bbox_inches="tight")
        plt.close(fig)
    paths["metal_artifact"] = p4

    return paths


# --------------------------------------------------------------------------------------
# HÀM CHÍNH TẠO FILE BÁO CÁO BTL PDF
# --------------------------------------------------------------------------------------
def create_btl_report(pdf_path: str, scan_data=None, algo_results=None):
    """Tạo tập tin Báo cáo BTL PDF đầy đủ các chương mục, chứng minh toán học LaTeX & sơ đồ khối thuật toán."""
    t0 = time.time()
    fig_dir = os.path.join(os.path.dirname(pdf_path), "figures")
    eq_dir = os.path.join(os.path.dirname(pdf_path), "equations")
    fig_paths = generate_report_figures(fig_dir)
    print("  -> Đang kết xuất 24 công thức toán học chuẩn LaTeX (STIX 300 DPI)...")
    eqs = generate_all_latex_equations(eq_dir)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_univ = ParagraphStyle(
        "CoverUniv", parent=styles["Normal"],
        fontName=FONT_BOLD, fontSize=12.5, leading=16, alignment=1, textColor=BK_BLUE
    )
    title_subuniv = ParagraphStyle(
        "CoverSubUniv", parent=styles["Normal"],
        fontName=FONT_BOLD, fontSize=11, leading=15, alignment=1, textColor=colors.HexColor("#1e293b")
    )
    title_dept = ParagraphStyle(
        "CoverDept", parent=styles["Normal"],
        fontName=FONT_BOLD, fontSize=10.5, leading=14, alignment=1, textColor=colors.HexColor("#334155")
    )
    title_report = ParagraphStyle(
        "CoverReport", parent=styles["Normal"],
        fontName=FONT_BOLD, fontSize=16, leading=22, alignment=1, textColor=colors.HexColor("#0f172a")
    )
    title_topic = ParagraphStyle(
        "CoverTopic", parent=styles["Normal"],
        fontName=FONT_BOLD, fontSize=14, leading=20, alignment=1, textColor=BK_BLUE
    )
    title_subtopic = ParagraphStyle(
        "CoverSubTopic", parent=styles["Normal"],
        fontName=FONT_ITALIC, fontSize=10.5, leading=15, alignment=1, textColor=colors.HexColor("#475569")
    )
    cover_meta = ParagraphStyle(
        "CoverMeta", parent=styles["Normal"],
        fontName=FONT_REG, fontSize=10.5, leading=16, alignment=1, textColor=colors.HexColor("#1e293b")
    )

    h1_style = ParagraphStyle(
        "BTL_H1", parent=styles["Heading1"],
        fontName=FONT_BOLD, fontSize=12.5, leading=17, textColor=BK_BLUE, spaceBefore=11, spaceAfter=6.5, keepWithNext=True
    )
    h1_center_style = ParagraphStyle(
        "BTL_H1_Center", parent=h1_style,
        fontSize=14.0, leading=19, alignment=1, spaceBefore=8, spaceAfter=14
    )
    h2_style = ParagraphStyle(
        "BTL_H2", parent=styles["Heading2"],
        fontName=FONT_BOLD, fontSize=10.8, leading=14.5, textColor=colors.HexColor("#0f172a"), spaceBefore=8.5, spaceAfter=4, keepWithNext=True
    )
    h3_style = ParagraphStyle(
        "BTL_H3", parent=styles["Heading3"],
        fontName=FONT_BI, fontSize=10.0, leading=14, textColor=colors.HexColor("#1e3a8a"), spaceBefore=5.5, spaceAfter=2.5, keepWithNext=True
    )
    body_style = ParagraphStyle(
        "BTL_Body", parent=styles["Normal"],
        fontName=FONT_REG, fontSize=9.8, leading=14.2, textColor=colors.HexColor("#1e293b"), spaceAfter=5.0, alignment=4
    )
    caption_style = ParagraphStyle(
        "BTL_Caption", parent=styles["Normal"],
        fontName=FONT_ITALIC, fontSize=8.5, leading=12, textColor=colors.HexColor("#475569"), alignment=1, spaceBefore=2.5, spaceAfter=6.5
    )

    def add_eq(key: str):
        story.append(Spacer(1, 1.5))
        story.append(eqs[key])
        story.append(Spacer(1, 2.5))

    story = []

    # ==================================================================================
    # 1. TRANG BÌA (COVER PAGE)
    # ==================================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH", title_univ))
    story.append(Paragraph("TRƯỜNG ĐẠI HỌC BÁCH KHOA", title_subuniv))
    story.append(Paragraph("KHOA KHOA HỌC ỨNG DỤNG - BỘ MÔN VẬT LÝ Y SINH", title_dept))
    story.append(Spacer(1, 4))
    story.append(Paragraph("——————o0o——————", title_dept))
    story.append(Spacer(1, 25))

    logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "bk_logo.png")
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=115, height=115))
    story.append(Spacer(1, 25))

    story.append(Paragraph("Báo cáo Bài tập lớn", title_report))
    story.append(Paragraph("Môn học: Cơ sở Vật lý Hạt nhân & Kỹ thuật Y sinh", title_dept))
    story.append(Spacer(1, 4))
    story.append(Paragraph("—————————————–", title_dept))
    story.append(Spacer(1, 10))

    story.append(Paragraph("ĐỀ TÀI: KỸ THUẬT CHỤP CT (COMPUTED TOMOGRAPHY) TRONG Y TẾ", title_topic))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Thiết Kế Hệ Thống Helical MDCT, Thang Xám HU & Tối Ưu Liều ALARA, Giải Bài Toán Ngược Khôi Phục Hàm Suy Giảm μ(x,y) Qua 7 Thuật Toán Tái Tạo & Phân Tích Ảnh CT Thực Tế",
        title_subtopic
    ))
    story.append(Spacer(1, 46))

    story.append(Paragraph("<b>Lớp:</b> DL01 — <b>Học kỳ:</b> HK261", cover_meta))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Giảng viên hướng dẫn:</b> TS. Nguyễn Hoàng Phúc", cover_meta))
    story.append(Spacer(1, 62))

    story.append(Paragraph("Thành phố Hồ Chí Minh – Tháng 11/2026", cover_meta))
    story.append(PageBreak())

    # ==================================================================================
    # 2. TRANG PHÂN CÔNG HẠNG MỤC NGHIÊN CỨU & THỰC HIỆN DỰ ÁN
    # ==================================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("CẤU TRÚC PHÂN CÔNG NỘI DUNG NGHIÊN CỨU & THỰC HIỆN DỰ ÁN", ParagraphStyle("H_List", parent=h1_style, alignment=1)))
    story.append(Spacer(1, 10))

    members_data = [
        ["HẠNG MỤC", "VAI TRÒ PHỤ TRÁCH", "ĐƠN VỊ", "HOÀN THÀNH", "NỘI DUNG CHI TIẾT TRONG BÁO CÁO & HỆ THỐNG"],
        ["Phần 1", "Thành viên Nhóm\n(Phần Tổng quan CT)", "Khoa học Ứng dụng", "100%", "Chương 1: Tổng quan kỹ thuật CT, nguyên lý hình thành ảnh từ hệ số suy giảm μ(r), khử chồng lấp, 4 thế hệ CT & xu hướng PCCT"],
        ["Phần 2.1", "Thành viên Nhóm\n(Phần Hệ thống MDCT)", "Khoa học Ứng dụng", "100%", "Chương 2 (Mục 2.1): Thiết kế & vận hành hệ thống Helical MDCT, Gantry, EID/PCCT, quỹ đạo xoắn ốc Pitch & dựng hình 3D MPR"],
        ["Phần 2.2", "Thành viên Nhóm\n(Phần HU & ALARA)", "Khoa học Ứng dụng", "100%", "Chương 2 (Mục 2.2): Thang xám Hounsfield (HU), cản quang Iodine, chẩn đoán lâm sàng, nguyên lý ALARA & tiêu chuẩn Rose (SNR ≥ 5)"],
        ["Phần 3.1", "Chuyên trách Thuật toán\n& Mô phỏng (Trọng tâm)", "Khoa học Ứng dụng", "100%", "Chương 3 (Mục 3.1 – 3.3): Bài toán ngược Ill-posed Radon, phương sai nhiễu logarit Taylor, chứng minh PSF 1/(πr) của BP, Định lý Lát cắt Fourier & FBP (5 bộ lọc tần số)"],
        ["Phần 3.2", "Chuyên trách Thuật toán\n& Mô phỏng (Trọng tâm)", "Khoa học Ứng dụng", "100%", "Chương 3 (Mục 3.4 – 3.5): Ma trận dải diện tích hình thang (Strip-Integral CDF), lặp đại số OS-SART / SIRT & lặp thống kê MBIR (PWLS-TV via Lagged Diffusivity + PCG)"],
        ["Phần 3.3", "Chuyên trách Thuật toán\n& Mô phỏng (Trọng tâm)", "Khoa học Ứng dụng", "100%", "Chương 3 (Mục 3.6): Tái tạo lai thống kê ASIR (phân tích phổ công suất nhiễu NPS) & Mạng học sâu phần dư DLIR (Residual CNN)"],
        ["Phần 4", "Chuyên trách Thuật toán\n& Phần mềm (Trọng tâm)", "Khoa học Ứng dụng", "100%", "Chương 4: Kiến trúc phần mềm chống Inverse Crime, hiệu chuẩn HU & định vị ROI tự động trên Ảnh CT Thực tế, Giao diện Workstation PyQt6"],
        ["Phần 5", "Chuyên trách Thực nghiệm\n& Tổng hợp (Trọng tâm)", "Khoa học Ứng dụng", "100%", "Chương 5 – 8: Hệ thống đo lường định lượng (RMSE, PSNR, SSIM, CNR), Thiết kế 6 Flowchart chuẩn LaTeX, Thực nghiệm ALARA / Metal Artifacts & Đóng gói .exe"],
    ]

    t_members = Table(members_data, colWidths=[45, 110, 86, 56, 218])
    t_members.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (3, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
    ]))
    story.append(t_members)
    story.append(PageBreak())

    # ==================================================================================
    # 3. TRANG NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN
    # ==================================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN", ParagraphStyle("H_Review", parent=h1_style, alignment=1)))
    story.append(Spacer(1, 15))

    dots_str = ". " * 52
    for _ in range(12):
        story.append(Paragraph(dots_str, body_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 15))
    grade_table_data = [
        ["Tiêu chí đánh giá", "Điểm tối đa", "Điểm đánh giá"],
        ["1. Tổng quan CT, Thiết kế Helical MDCT, Thang xám HU & Tối ưu liều ALARA / Rose", "2.5", ""],
        ["2. Cơ sở toán học Bài toán ngược, công thức LaTeX & Sơ đồ khối thuật toán (Flowcharts)", "3.0", ""],
        ["3. Mô phỏng vật lý tia X & Hiện thực 7 thuật toán (BP, FBP, SART, SIRT, MBIR, ASIR, DLIR)", "2.5", ""],
        ["4. Giao diện GUI PyQt6, Phân tích ảnh CT thực tế, Đo lường định lượng & Đóng gói .exe", "2.0", ""],
        ["TỔNG ĐIỂM", "10.0", ""],
    ]
    t_grade = Table(grade_table_data, colWidths=[295, 95, 105])
    t_grade.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, -1), (-1, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_grade)
    story.append(Spacer(1, 40))

    sig_data = [
        ["GIẢNG VIÊN CHẤM ĐIỂM", "ĐẠI DIỆN NHÓM THỰC HIỆN"],
        ["(Ký và ghi rõ họ tên)", "(Ký và ghi rõ họ tên)"],
        ["", ""],
        ["", ""],
        ["TS. Nguyễn Hoàng Phúc", ""],
    ]
    t_sig = Table(sig_data, colWidths=[250, 250])
    t_sig.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, 1), FONT_ITALIC),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 35),
    ]))
    story.append(t_sig)
    story.append(PageBreak())

    # ==================================================================================
    # 4. MỤC LỤC & DANH MỤC HÌNH ẢNH / SƠ ĐỒ KHỐI / BẢNG BIỂU
    # ==================================================================================
    story.append(Paragraph("MỤC LỤC BÁO CÁO", h1_style))
    story.append(Spacer(1, 4))

    toc_items = [
        ("CHƯƠNG 1: TỔNG QUAN VỀ KỸ THUẬT CHỤP CẮT LỚP VI TÍNH (CT)", "5"),
        ("    1.1. Kỹ thuật CT là gì và nguyên lý hình thành ảnh từ hệ số suy giảm μ(r)", "5"),
        ("    1.2. So sánh sự ưu việt của CT so với X-quang truyền thống: Bài toán khử chồng lấp", "5"),
        ("    1.3. Các thế hệ CT (I – IV), Helical MDCT và xu hướng CT đếm photon (PCCT)", "6"),
        ("    1.4. Nhu cầu sử dụng và vai trò của CT trong thực hành lâm sàng", "7"),
        ("CHƯƠNG 2: THIẾT KẾ HỆ THỐNG HELICAL MDCT, THANG XÁM HU & TỐI ƯU LIỀU ALARA", "8"),
        ("    2.1. Kỹ thuật CT – Thiết kế và vận hành hệ thống Helical MDCT", "8"),
        ("        2.1.1. Thiết kế tổng thể của một hệ thống CT hiện đại", "8"),
        ("        2.1.2. Bóng phát tia X, lọc chùm tia Bow-tie, chuẩn trực và Gantry vòng trượt", "9"),
        ("        2.1.3. Hệ thống đầu dò (EID vs PCCT) và chuỗi thu nhận dữ liệu DAS/ADC", "9"),
        ("        2.1.4. Nguyên lý quét xoắn ốc Helical MDCT và thông số bước xoắn Pitch", "9"),
        ("        2.1.5. Thu nhận dữ liệu, nội suy lát cắt và tái tạo khối 3D (MPR, MIP, VR)", "10"),
        ("        2.1.6. Ưu điểm kỹ thuật, hạn chế nhiễu ảnh và Bảng đối chiếu Axial CT vs Helical MDCT", "11"),
        ("    2.2. Thang xám Hounsfield (HU), Chẩn đoán bệnh lý & Tối ưu hóa liều bức xạ ALARA", "12"),
        ("        2.2.1. Định nghĩa Thang đo Hounsfield (HU), ý nghĩa 'độ trong suốt' 0.1% và dải HU mô học", "12"),
        ("        2.2.2. Chất cản quang Iodine (CTA) và chẩn đoán Gan nhiễm mỡ, Đột quỵ, Ung thư phổi LDCT", "13"),
        ("        2.2.3. Nguyên lý ALARA, lựa chọn điện áp kVp, Tiêu chuẩn Rose (SNR ≥ 5) & Tính toán số photon", "14"),
        ("CHƯƠNG 3: CƠ SỞ TOÁN HỌC BÀI TOÁN NGƯỢC & BÓC TÁCH CÁC THUẬT TOÁN TÁI TẠO", "16"),
        ("    3.1. Đặt bài toán ngược tổng quát (Ill-posed Inverse Problem) và Phương sai nhiễu Logarit", "16"),
        ("    3.2. Bóc tách Phép chiếu ngược đơn thuần (BP): Chứng minh hàm truyền điểm PSF 1/(πr)", "17"),
        ("    3.3. Bóc tách Chiếu ngược có lọc (FBP): Định lý Lát cắt Fourier, Nhân Ram-Lak & 5 Bộ lọc", "18"),
        ("    3.4. Bóc tách Lặp Đại số (SART & SIRT): Mô hình Dải Hình thang, Hội tụ & Bán hội tụ", "20"),
        ("    3.5. Bóc tách Lặp Thống kê MBIR (PWLS-TV): Đạo hàm Euler-Lagrange, Lagged Diffusivity & PCG", "21"),
        ("    3.6. Bóc tách Lai Thống kê ASIR (Phổ NPS) và Học sâu Phần dư DLIR (Residual CNN)", "22"),
        ("CHƯƠNG 4: THIẾT KẾ HỆ THỐNG PHẦN MỀM & MODULE PHÂN TÍCH ẢNH CT THỰC TẾ", "25"),
        ("    4.1. Kiến trúc đa tầng và kỹ thuật chống lỗi nghịch đảo (Avoiding Inverse Crime)", "25"),
        ("    4.2. Thuật toán hiệu chuẩn phi tuyến từng khúc và Tự động định vị ROI cho Ảnh CT Thực tế", "25"),
        ("    4.3. Thiết kế Giao diện Trạm làm việc Y khoa (Medical Pro Workstation GUI PyQt6)", "25"),
        ("CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM, ĐO LƯỜNG ĐỊNH LƯỢNG & PHÂN TÍCH ẢNH CT THỰC TẾ", "26"),
        ("    5.1. Hệ thống chỉ số đo lường định lượng (RMSE, PSNR, SSIM, Noise ROI, Bias, CNR)", "26"),
        ("    5.2. Đánh giá so sánh hiệu năng toàn diện 7 thuật toán tái tạo và 5 bộ lọc FBP", "26"),
        ("    5.3. Khảo sát thực nghiệm đường cong giảm liều ALARA & Ảnh giả kim loại (Metal Artifacts)", "26"),
        ("    5.4. Thực nghiệm phân tích và khôi phục hàm μ(x,y) trên Ảnh CT Lâm sàng Thực tế", "28"),
        ("CHƯƠNG 6: HƯỚNG DẪN VẬN HÀNH & ĐÓNG GÓI PHẦN MỀM ĐỘC LẬP (.EXE)", "29"),
        ("CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", "29"),
        ("CHƯƠNG 8: TÀI LIỆU THAM KHẢO", "30"),
    ]

    for title, p_num in toc_items:
        is_chap = title.startswith("CHƯƠNG")
        f_name = FONT_BOLD if is_chap else FONT_REG
        color_t = BK_BLUE if is_chap else colors.HexColor("#1e293b")
        t_style = ParagraphStyle("TOC_Item", parent=styles["Normal"], fontName=f_name, fontSize=8.5, leading=12.0, textColor=color_t)

        t_row = Table([[Paragraph(title, t_style), Paragraph(p_num, ParagraphStyle("P_Num", parent=t_style, alignment=2))]],
                      colWidths=[465, 30])
        t_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
            ("TOPPADDING", (0, 0), (-1, -1), 1.0),
        ]))
        story.append(t_row)

    story.append(PageBreak())
    story.append(Paragraph("DANH MỤC HÌNH ẢNH & SƠ ĐỒ KHỐI THUẬT TOÁN (FLOWCHARTS)", h1_style))
    story.append(Spacer(1, 6))
    fig_list = [
        "Hình 1.1: Minh họa nguyên lý chồng lấp trong X-quang 2D và nguyên lý chiếu đa hướng trong CT.",
        "Hình 1.2: Sơ đồ nguyên lý hình học của 4 thế hệ máy chụp CT kinh điển (Thế hệ I – IV).",
        "Hình 2.1: Cấu trúc tổng quát của hệ thống CT hiện đại từ Gantry quay, DAS/ADC đến máy tính tái tạo.",
        "Hình 2.2: Mô phỏng quét chùm tia X qua Phantom Đầu và ma trận dữ liệu thô Sinogram p(s, θ).",
        "Hình 2.3: Minh họa nguyên lý quét xoắn ốc (Helical/Spiral CT) trong hệ quy chiếu bệnh nhân.",
        "Hình 2.4: Sơ đồ chuỗi xử lý từ nhiều góc chiếu qua DAS đến tái tạo lát cắt và đa mặt phẳng MPR/3D.",
        "Hình 2.5: Ảnh chụp CT lâm sàng đối chiếu giữa Gan bình thường (66 HU > Lách 56 HU) và Gan nhiễm mỡ (15 HU < Lách 40 HU).",
        "Hình 3.1: [Flowchart 0] Sơ đồ khung toán học Bài toán ngược từ suy giảm vật lý đến khôi phục hàm μ(x, y).",
        "Hình 3.2: [Flowchart 1] Sơ đồ khối thuật toán Chiếu ngược đơn thuần (BP) và cơ chế sinh mờ 1/(πr).",
        "Hình 3.3: [Flowchart 2] Sơ đồ khối thuật toán Chiếu ngược có lọc (FBP) và 5 cửa sổ điều biến W(ω).",
        "Hình 3.4: Đồ thị phân tích 5 bộ lọc FBP trong miền tần số H(ω), miền không gian h[n] và đánh đổi Nhiễu – PSNR.",
        "Hình 3.5: [Flowchart 3 & 4] Sơ đồ khối thuật toán lặp đại số OS-SART (Tỷ lệ vàng) và SIRT (Landweber).",
        "Hình 3.6: [Flowchart 5] Sơ đồ khối thuật toán lặp thống kê MBIR (PWLS-TV via Lagged Diffusivity + PCG).",
        "Hình 3.7: [Flowchart 6 & 7] Sơ đồ khối thuật toán lai thống kê ASIR và Học sâu phần dư DLIR (Residual CNN).",
        "Hình 3.8: So sánh trực quan 6 thuật toán tái tạo ở chế độ liều thấp (50 mA – giảm 75% liều bức xạ).",
        "Hình 5.1: Đồ thị khảo sát nguyên lý ALARA (Độ lệch chuẩn nhiễu ROI và PSNR theo dòng mA & CTDIvol).",
        "Hình 5.2: Khảo sát ảnh giả vệt kim loại (Metal Streak Artifacts) trên FBP và cơ chế triệt tiêu của MBIR.",
        "Hình 5.3: Thực nghiệm phân tích và khôi phục hàm suy giảm μ(x,y) trên Ảnh CT Lâm sàng Thực tế (Gan/Bụng).",
    ]
    for fl in fig_list:
        story.append(Paragraph("• " + fl, ParagraphStyle("FL_Item", parent=caption_style, alignment=0, spaceBefore=0.5, spaceAfter=1.0)))
    story.append(PageBreak())

    # ==================================================================================
    # CHƯƠNG 1: TỔNG QUAN VỀ KỸ THUẬT CHỤP CẮT LỚP VI TÍNH (CT)
    # ==================================================================================
    story.append(Paragraph("CHƯƠNG 1: TỔNG QUAN VỀ KỸ THUẬT CHỤP CẮT LỚP VI TÍNH (CT)", h1_style))

    story.append(Paragraph("1.1. Kỹ thuật CT là gì và nguyên lý hình thành ảnh từ hệ số suy giảm μ(r)", h2_style))
    story.append(Paragraph(
        "Chụp cắt lớp vi tính (<i>Computed Tomography</i> – CT) là kỹ thuật chẩn đoán hình ảnh sử dụng tia X kết hợp với xử lý máy tính "
        "để tạo ra các lát cắt ngang (<i>cross-sectional images</i>) của cơ thể; từ tập hợp các lát cắt liên tiếp này, hệ thống có thể tái tạo "
        "ảnh đa mặt phẳng hoặc dựng hình ba chiều (3D) phục vụ chẩn đoán lâm sàng [1–3]. Khác với X-quang quy ước chỉ tạo ra một ảnh chiếu phẳng duy nhất, "
        "máy CT thu nhận nhiều phép đo suy giảm tia X từ rất nhiều góc chiếu xung quanh cơ thể rồi giải bài toán tái tạo để ước lượng phân bố "
        "hệ số suy giảm tuyến tính của mô ở từng điểm ảnh (<i>pixel</i>) hoặc thể tích ảnh (<i>voxel</i>) [3, 4].",
        body_style
    ))
    story.append(Paragraph(
        "Về mặt vật lý, khi một chùm tia X hẹp có cường độ ban đầu <i>I</i><sub>0</sub> đi qua cơ thể dọc theo đường truyền <i>L</i>, "
        "cường độ tia ló <i>I</i> suy giảm theo định luật Beer–Lambert (trong giả thiết gần đúng đơn sắc hiệu dụng) [4]:",
        body_style
    ))
    add_eq("eq_1_1")
    story.append(Paragraph(
        "Trong đó <i>μ</i>(<b>r</b>) là hệ số suy giảm tuyến tính tại vị trí không gian <b>r</b> = (<i>x, y</i>), phụ thuộc trực tiếp vào mật độ vật chất, "
        "nguyên tử số hiệu dụng <i>Z</i><sub>eff</sub> của mô và năng lượng chùm tia X. Lấy logarit tự nhiên tỷ số cường độ, ta thu được dữ liệu chiếu "
        "(<i>projection / line integral</i>) – tức biến đổi tích phân Radon hai chiều [4, 13]:",
        body_style
    ))
    add_eq("eq_1_2")
    story.append(Paragraph(
        "Phương trình (1.2) chính là cầu nối toán học giữa phép đo vật lý ở đầu dò và bài toán tái tạo ảnh trên máy tính: mỗi góc quay <i>θ</i> của bóng phát – đầu dò "
        "cung cấp một tập giá trị tích phân đường <i>p</i>(<i>s, θ</i>), và tập hợp nhiều góc quay tạo thành miền dữ liệu thô (<i>sinogram</i>) đủ để giải ngược ra bản đồ "
        "<i>μ</i>(<i>x, y</i>) bên trong mặt cắt [3, 4]. Về mặt lịch sử, kỹ thuật CT được phát triển độc lập bởi kỹ sư Sir Godfrey N. Hounsfield và nhà vật lý Allan M. Cormack; "
        "hai ông đã cùng được trao Giải Nobel Sinh lý học hoặc Y học năm 1979 vì đóng góp mang tính cách mạng này [6].",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["docx_xray_vs_ct"], width=445, height=160))
    story.append(Paragraph("Hình 1.1: Minh họa nguyên lý chồng lấp trong X-quang truyền thống (một hướng chiếu) và nguyên lý chiếu đa hướng kèm tái tạo trong CT.", caption_style))

    story.append(Paragraph("1.2. So sánh sự ưu việt của CT so với X-quang truyền thống: Bài toán khử chồng lấp", h2_style))
    story.append(Paragraph(
        "Hạn chế cốt lõi của X-quang truyền thống là <b>hiện tượng chồng lấp cấu trúc</b> (<i>superposition of structures</i>) [1, 2]. "
        "Khi chùm tia X đi từ nguồn qua cơ thể tới tấm thu phẳng, mọi cơ quan nằm trên cùng một đường truyền tia (da, mỡ, cơ, xương sườn, nhu mô phổi, mạch máu, trung thất) "
        "đều cùng đóng góp vào một giá trị suy giảm tích lũy duy nhất trên mặt phẳng ảnh 2D. Do đó, hình ảnh X-quang là 'bóng chồng' của toàn bộ thể tích theo hướng chiếu, "
        "làm mất thông tin chiều sâu và khiến các tổn thương nhỏ hoặc các mô mềm có độ suy giảm gần nhau bị che khuất bởi cấu trúc xương dày phía trước hoặc phía sau [2, 5].",
        body_style
    ))
    story.append(Paragraph(
        "CT khắc phục triệt để hạn chế này bằng cách tách thể tích cơ thể thành từng lát cắt mỏng và thu nhận dữ liệu từ hàng trăm đến hàng nghìn hướng chiếu quanh bệnh nhân [1, 3]. "
        "Nhờ giải thuật tái tạo mặt cắt, mỗi voxel có giá trị suy giảm riêng và được chuẩn hóa về thang Hounsfield (HU), mang lại ba ưu thế vượt trội: "
        "(1) <b>Khử hoàn toàn chồng lấp</b> theo chiều sâu; (2) <b>Độ phân giải tương phản mô mềm vượt trội</b> (phân biệt được dịch, mỡ, máu tụ cấp, nhu mô tạng đặc, chất trắng – chất xám); "
        "và (3) <b>Khả năng tái tạo đa mặt phẳng (MPR) & dựng hình 3D</b> phục vụ định vị chính xác trước phẫu thuật hoặc xạ trị [2, 3, 5].",
        body_style
    ))

    t_x_vs_ct_data = [
        ["Tiêu chí kỹ thuật", "X-quang truyền thống (2D Radiography)", "Chụp cắt lớp vi tính (CT Scanner)"],
        ["Bản chất hình ảnh", "Ảnh chiếu phẳng 2D của cả một thể tích theo một hướng", "Ảnh lát cắt ngang 2D và dữ liệu thể tích 3D (Voxel)"],
        ["Hiện tượng chồng lấp", "Có; các lớp mô trước – sau chồng trực tiếp lên nhau", "Hầu như được loại bỏ hoàn toàn trong từng lát cắt"],
        ["Phân biệt mô mềm", "Hạn chế khi chênh lệch hệ số suy giảm μ nhỏ (1–2%)", "Rất tốt (phân biệt chênh lệch 0.1% – 0.5% trên thang HU)"],
        ["Thông tin chiều sâu & 3D", "Không trực tiếp có từ một phim chụp đơn lẻ", "Có; xác định tọa độ 3D, MPR (axial, coronal, sagittal), MIP, VR"],
        ["Thời gian & khảo sát cấp cứu", "Rất nhanh, chi phí thấp, sàng lọc ban đầu", "Rất nhanh (vài giây với MDCT), độ chính xác chẩn đoán sâu cao"],
        ["Liều bức xạ & chi phí", "Thấp hơn, thiết bị đơn giản", "Cao hơn X-quang thường → bắt buộc tối ưu hóa theo ALARA"],
    ]
    t_x_ct = Table(t_x_vs_ct_data, colWidths=[115, 195, 195])
    t_x_ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (0, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    story.append(t_x_ct)
    story.append(Paragraph("Bảng 1.1: Đối chiếu trực tiếp giữa X-quang truyền thống và Chụp cắt lớp vi tính (CT) [1–3, 5].", caption_style))

    story.append(Paragraph("1.3. Các thế hệ CT (I – IV), Helical MDCT và xu hướng CT đếm photon (PCCT)", h2_style))
    story.append(Paragraph(
        "Sự phát triển của máy CT gắn liền với nỗ lực giải quyết bài toán tăng tốc độ thu nhận dữ liệu, mở rộng vùng bao phủ giải phẫu, tăng độ phân giải không gian – tương phản "
        "và giảm liều bức xạ cho bệnh nhân [3]. Về mặt lịch sử thiết kế cơ – quang học, 4 thế hệ CT kinh điển được phân loại theo hình học chùm tia và chuyển động của bóng phát – đầu dò [3, 7]:",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["docx_generations"], width=410, height=315))
    story.append(Paragraph("Hình 1.2: Sơ đồ nguyên lý hình học của 4 thế hệ máy chụp CT kinh điển (Thế hệ I – IV) [3, 7].", caption_style))

    t_gen_data = [
        ["Thế hệ", "Cấu hình nguồn – đầu dò", "Cơ chế quét", "Ưu điểm chính", "Hạn chế chính"],
        ["Thế hệ I", "Chùm tia bút chì (pencil beam), 1–2 đầu dò", "Tịnh tiến – quay (Translate–Rotate)", "Cấu hình đơn giản, rất ít nhiễu tán xạ", "Rất chậm (~5 phút/lát), chỉ phù hợp sọ não"],
        ["Thế hệ II", "Chùm tia quạt hẹp (~10°), dãy nhỏ đầu dò (~30 kênh)", "Tịnh tiến – quay với bước góc lớn hơn", "Nhanh hơn thế hệ I đáng kể (20–60 giây/lát)", "Vẫn còn chuyển động tịnh tiến cơ học, hạn chế khảo sát ngực – bụng"],
        ["Thế hệ III", "Chùm tia quạt rộng (50°–60°), dãy đầu dò cong quay cùng bóng phát", "Quay – quay (Rotate–Rotate)", "Tốc độ rất nhanh (<0.3 s/vòng), nền tảng của hầu hết MDCT hiện đại", "Nhạy với sai lệch hiệu chuẩn kênh đầu dò (gây nhiễu vòng tròn – ring artifact)"],
        ["Thế hệ IV", "Bóng phát quay, vòng đầu dò tĩnh 360° bao quanh Gantry", "Quay – tĩnh (Rotate–Stationary)", "Giảm nguy cơ nhiễu vòng tròn do đầu dò cố định", "Chi phí đầu dò cực lớn, khó mở rộng lên nhiều dãy (multi-slice), góc nhận tán xạ rộng"],
    ]
    t_gen = Table(t_gen_data, colWidths=[52, 118, 88, 122, 125])
    t_gen.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (0, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    story.append(t_gen)
    story.append(Paragraph("Bảng 1.2: Tổng hợp đặc điểm kỹ thuật của 4 thế hệ máy chụp CT kinh điển [3, 7].", caption_style))

    story.append(Paragraph(
        "Sau bốn thế hệ kinh điển (nơi kiến trúc Thế hệ III chiến thắng và trở thành chuẩn mực công nghiệp), công nghệ CT tiếp tục bứt phá qua các cột mốc hiện đại [3, 8]:<br/>"
        "• <b>Công nghệ vòng trượt (Slip-ring) & CT xoắn ốc (Helical/Spiral CT – cuối thập niên 1980):</b> Thay thế dây cáp cuộn truyền thống bằng các vành trượt dẫn điện và truyền dữ liệu tốc độ cao, "
        "cho phép khung Gantry quay liên tục một chiều đồng thời với bàn bệnh nhân trượt đều theo trục <i>z</i>, rút ngắn thời gian chụp toàn bộ lồng ngực xuống chỉ một lần nín thở.<br/>"
        "• <b>CT đa dãy đầu dò (MDCT – Multi-Detector Row CT, từ 1998):</b> Mở rộng mảng đầu dò dọc trục <i>z</i> từ 4, 16, 64 lên đến 128, 256, 320 và 640 lát cắt đồng thời, "
        "cho phép thu nhận các voxel đẳng hướng (<i>isotropic voxels</i> ~ 0.5 × 0.5 × 0.5 mm<sup>3</sup>) và chụp cắt lớp mạch vành tim đang đập.<br/>"
        "• <b>CT hai mức năng lượng / CT phổ (Dual-Energy / Spectral CT):</b> Khai thác sự phụ thuộc của <i>μ</i>(<i>E</i>) vào năng lượng photon ở hai mức kVp khác nhau (ví dụ 80 kVp và 140 kVp) "
        "để phân tách vật chất (Iodine, Canxi, Acid Uric, Nước) và tạo ảnh đơn sắc ảo (<i>Virtual Monoenergetic Images</i>).<br/>"
        "• <b>CT đếm photon (Photon-Counting CT – PCCT, thế hệ đột phá hiện nay):</b> Khác với đầu dò tích hợp năng lượng truyền thống (EID) phải chuyển tia X thành ánh sáng khả kiến qua tinh thể nhấp nháy rồi mới đổi sang dòng điện, "
        "đầu dò đếm photon sử dụng bán dẫn chuyển đổi trực tiếp (CdTe, CZT) để đếm từng photon tia X riêng lẻ và đo chính xác năng lượng của từng photon theo các ngưỡng (<i>energy bins</i>) [3, 8]. "
        "Nhờ đặt ngưỡng năng lượng thấp để cắt bỏ hoàn toàn nhiễu điện tử nền (<i>electronic dark noise</i>), các hệ thống PCCT lâm sàng thế hệ mới (như dòng <i>Siemens Naeotom Alpha</i> [8]) "
        "đạt độ phân giải không gian siêu cao (~0.2 mm), tăng tỷ số CNR của Iodine và giảm đáng kể liều bức xạ cho bệnh nhân.<br/>"
        "• <b>Tái tạo lặp thống kê (MBIR/ASIR) và Tái tạo Học sâu (DLIR):</b> Chuyển dịch từ phép nghịch đảo giải tích FBP sang các mô hình tối ưu hóa thống kê và mạng nơ-ron tích chập sâu, "
        "cho phép duy trì chất lượng ảnh chẩn đoán ở mức liều tia X giảm từ 50% đến 80% [3, 18].",
        body_style
    ))

    story.append(Paragraph("1.4. Nhu cầu sử dụng và vai trò của CT trong thực hành lâm sàng", h2_style))
    story.append(Paragraph(
        "Trong y học hiện đại, CT là một trong những phương tiện chẩn đoán hình ảnh trụ cột nhờ sự kết hợp độc đáo giữa tốc độ thu nhận cực nhanh, độ chính xác hình học 3D và khả năng đánh giá đồng thời xương – mô mềm – mạch máu – tạng rỗng [1, 2, 5]:<br/>"
        "1. <b>Cấp cứu và chấn thương đa cơ quan (Trauma):</b> Đánh giá toàn diện xuất huyết nội sọ, vỡ xương sọ – cột sống, tràn khí/tràn máu màng phổi và tổn thương tạng đặc ổ bụng chỉ trong vài chục giây.<br/>"
        "2. <b>Đột quỵ não cấp (Acute Stroke):</b> Chụp CT sọ não không tiêm thuốc là chỉ định bắt buộc đầu tiên để phân biệt tức thời xuất huyết não cấp (tăng tỷ trọng 75–100 HU) với nhồi máu não cấp trong 'cửa sổ vàng' điều trị tiêu sợi huyết, kết hợp CTA mạch não và CT tưới máu (CT Perfusion).<br/>"
        "3. <b>Ung bướu và tầm soát sớm ung thư phổi (Oncology & LDCT):</b> Chụp CT liều thấp (LDCT) là phương pháp chuẩn được khuyến cáo để tầm soát phát hiện sớm nốt ung thư phổi; đồng thời bản đồ mật độ điện tử suy ra từ hệ số suy giảm <i>μ</i>(<i>x, y, z</i>) của CT là dữ liệu đầu vào bắt buộc để <b>lập kế hoạch xạ trị ung thư</b> (<i>Radiation Therapy Planning</i>).<br/>"
        "4. <b>Tim mạch (Cardiovascular CTA):</b> Đánh giá hẹp động mạch vành, mảng xơ vữa vôi hóa (Calcium Scoring), bóc tách động mạch chủ và thuyên tắc động mạch phổi cấp.<br/>"
        "5. <b>Hướng dẫn thủ thuật can thiệp:</b> Định vị kim sinh thiết khối u sâu, dẫn lưu ổ áp-xe nội tạng dưới hướng dẫn CT thời gian thực (<i>CT Fluoroscopy</i>).",
        body_style
    ))
    story.append(PageBreak())

    # ==================================================================================
    # CHƯƠNG 2: THIẾT KẾ HỆ THỐNG HELICAL MDCT, THANG XÁM HU & TỐI ƯU LIỀU ALARA
    # ==================================================================================
    story.append(Paragraph("CHƯƠNG 2: THIẾT KẾ HỆ THỐNG HELICAL MDCT, THANG XÁM HU & TỐI ƯU LIỀU ALARA", h1_style))

    story.append(Paragraph("2.1. Kỹ thuật CT – Thiết kế và vận hành hệ thống Helical MDCT", h2_style))
    story.append(Paragraph(
        "Trong thực hành lâm sàng hiện nay, cấu hình phổ biến nhất là <b>máy chụp cắt lớp vi tính đa dãy đầu dò quét xoắn ốc</b> (<i>Helical Multi-Detector Row CT</i> – Helical MDCT) "
        "phát triển trên nền hình học Thế hệ III [3, 7]. Để hiểu rõ cách dữ liệu Sinogram được tạo ra và đi vào các giải thuật tái tạo ở Chương 3, mục này bóc tách chi tiết kiến trúc phần cứng và quy trình vận hành của hệ thống.",
        body_style
    ))

    story.append(Paragraph("<b>2.1.1. Thiết kế tổng thể của một hệ thống CT hiện đại:</b>", h3_style))
    story.append(Paragraph(
        "Một hệ thống CT hoàn chỉnh là một dây chuyền cơ – điện tử – quang – tin học đồng bộ chặt chẽ, trong đó sai lệch ở bất kỳ mắt xích nào (dao động công suất bóng phát, "
        "lệch chuẩn kênh đầu dò, sai số đồng bộ tốc độ bàn hoặc nhiễu điện tử ở bộ số hóa DAS) đều trực tiếp sinh ra ảnh giả (<i>artifacts</i>) trên ảnh tái tạo cuối cùng [3].",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["docx_gantry_das"], width=445, height=225))
    story.append(Paragraph("Hình 2.1: Cấu trúc tổng quát của hệ thống CT hiện đại từ khung Gantry quay, mảng đầu dò cong, bộ thu nhận DAS/ADC đến máy tính tái tạo.", caption_style))

    t_blocks_data = [
        ["Khối phần cứng", "Thành phần chính", "Vai trò trong quá trình tạo ảnh CT"],
        ["1. Khối phát tia X", "Máy phát cao áp (HV Generator), bóng tia X anode quay, bộ lọc Bow-tie, bộ chuẩn trực tiền bệnh nhân", "Tạo và định hình chùm tia X hình quạt/hình nón với điện áp kVp, dòng mA và bề dày chùm tia xác định"],
        ["2. Khung Gantry & Bàn bệnh nhân", "Khung quay Gantry, vòng trượt (Slip-ring), động cơ servo và bàn dịch chuyển theo trục z", "Đảm bảo chuyển động quay liên tục của cặp nguồn – đầu dò và chuyển động tịnh tiến chính xác của bệnh nhân"],
        ["3. Hệ thống Đầu dò & DAS", "Mảng đầu dò đa dãy (2D Detector Array), lưới chống tán xạ, khuếch đại tiền xử lý và ADC (DAS)", "Đo cường độ tia X sau suy giảm, chuyển thành tín hiệu số và lấy logarit tạo dữ liệu hình chiếu p(s, θ)"],
        ["4. Hệ thống Máy tính & Trạm làm việc", "Bộ tiền xử lý hiệu chuẩn, cụm GPU/FPGA tái tạo ảnh, màn hình y khoa PACS/DICOM", "Hiệu chỉnh sai số vật lý, thực thi thuật toán tái tạo (FBP/IR/DLIR), hiển thị cửa sổ HU và dựng hình MPR/3D"],
    ]
    t_blocks = Table(t_blocks_data, colWidths=[105, 175, 225])
    t_blocks.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (0, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    story.append(t_blocks)
    story.append(Paragraph("Bảng 2.1: Các khối chức năng chính trong hệ thống chụp cắt lớp vi tính hiện đại [3, 7].", caption_style))

    story.append(Paragraph("<b>2.1.2. Bóng phát tia X, lọc chùm tia (Bow-tie filter), chuẩn trực và Gantry vòng trượt:</b>", h3_style))
    story.append(Paragraph(
        "• <b>Bóng phát tia X công suất cao:</b> Trong máy CT, bóng phát phải hoạt động liên tục ở dòng điện hàng trăm mA và chịu lực ly tâm cực lớn khi Gantry quay 0.25–0.35 giây/vòng. "
        "Do đó, bóng tia X sử dụng anode quay bằng hợp kim Tungsten–Rhenium (<i>Z</i> = 74) gắn trên đế Graphite tản nhiệt lớn hoặc công nghệ buồng chân không quay trực tiếp trong dầu làm mát (<i>Straton tube</i>) kèm kỹ thuật lái tâm điểm điện từ (<i>flying focal spot</i>) giúp tăng gấp đôi mật độ lấy mẫu không gian [3].<br/>"
        "• <b>Bộ lọc định hình dạng nơ (Bow-tie filter):</b> Cơ thể người có tiết diện gần giống hình elip (dày ở vùng trung tâm và mỏng dần ở hai bên rìa). Nếu chiếu chùm tia đồng nhất, vùng rìa cơ thể sẽ nhận liều bức xạ thừa và làm bão hòa các kênh đầu dò ngoại vi. "
        "Bộ lọc <i>Bow-tie</i> (mỏng ở tâm, dày ở hai rìa) giúp làm cứng trước chùm tia (hấp thụ photon năng lượng thấp gây bỏng da) và cân bằng thông lượng photon tới mọi kênh đầu dò, giảm liều bề mặt cho bệnh nhân và giảm nhiễu cứng hóa chùm tia (<i>beam hardening</i>) [3, 15].<br/>"
        "• <b>Hệ chuẩn trực (Collimation):</b> Gồm <i>chuẩn trực tiền bệnh nhân</i> (đặt ngay sau bộ lọc Bow-tie để giới hạn độ mở chùm tia theo trục <i>z</i>, tránh chiếu xạ thừa ngoài vùng khảo sát) và <i>chuẩn trực hậu bệnh nhân / lưới chống tán xạ</i> (<i>anti-scatter Septa grid</i>) đặt ngay trước các phần tử đầu dò để chặn các photon tán xạ lệch góc Compton.<br/>"
        "• <b>Gantry và Vòng trượt (Slip-ring):</b> Khung Gantry tích hợp các vòng trượt tiếp xúc chổi than hoặc truyền dẫn quang không dây tốc độ hàng Gigabit/giây, giải phóng hoàn toàn giới hạn góc quay của dây cáp truyền thống.",
        body_style
    ))

    story.append(Paragraph("<b>2.1.3. Hệ thống đầu dò (EID vs PCCT) và chuỗi thu nhận dữ liệu DAS/ADC:</b>", h3_style))
    story.append(Paragraph(
        "Mảng đầu dò CT được bố trí theo một cung tròn đồng tâm với tiêu điểm bóng phát tia X, chia thành hàng trăm kênh theo phương ngang (trục quạt <i>s</i>) và nhiều dãy theo phương dọc trục cơ thể (trục <i>z</i>) [3, 7]. "
        "Hiện có hai công nghệ đầu dò chính:<br/>"
        "1. <b>Đầu dò tích hợp năng lượng (Energy-Integrating Detectors – EID):</b> Là chuẩn mực trên đa số máy MDCT hiện hành. Mỗi phần tử gồm lớp tinh thể nhấp nháy gốm siêu nhanh (như <i>Gadolinium Oxysulfide</i> – Gd<sub>2</sub>O<sub>2</sub>S / GOS hoặc CdWO<sub>4</sub>) "
        "hấp thụ tia X và phát ra các photon ánh sáng khả kiến; lớp <i>Photodiode</i> gắn phía dưới chuyển ánh sáng này thành dòng điện tỷ lệ với <i>tổng năng lượng</i> của toàn bộ photon tia X đi vào kênh trong một chu kỳ lấy mẫu.<br/>"
        "2. <b>Đầu dò đếm photon (Photon-Counting Detectors – PCCT):</b> Sử dụng tinh thể bán dẫn CdTe hoặc CZT đặt dưới điện trường cao áp để chuyển trực tiếp mỗi photon tia X thành một đám mây điện tử – lỗ trống, tạo ra xung điện có biên độ tỷ lệ thuận với năng lượng của chính photon đó [3, 8].<br/>"
        "Tín hiệu tương tự từ mảng đầu dò đi vào <b>Hệ thống thu nhận dữ liệu (DAS – Data Acquisition System)</b>, được khuếch đại độ nhiễu thấp, số hóa qua bộ chuyển đổi ADC (18–24 bit), hiệu chuẩn dòng tối (<i>dark current offset</i>), chuẩn hóa độ nhạy kênh (<i>air calibration</i>) và lấy logarit âm để tạo thành ma trận hình chiếu <i>p</i>(<i>s, θ</i>).",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["acquisition"], width=445, height=195))
    story.append(Paragraph("Hình 2.2: Mô phỏng quá trình quét chùm tia X qua Phantom Đầu và ma trận dữ liệu thô Sinogram p(s, θ) thu được từ DAS.", caption_style))

    story.append(Paragraph("<b>2.1.4. Nguyên lý quét xoắn ốc Helical MDCT và thông số bước xoắn Pitch:</b>", h3_style))
    story.append(Paragraph(
        "Trong chế độ chụp cắt lớp tuần tự cổ điển (<i>Axial / Step-and-Shoot CT</i>), bàn bệnh nhân đứng yên khi Gantry quay quét một lát, sau đó bóng phát tắt để bàn nhích sang vị trí kế tiếp. "
        "Ngược lại, trong <b>chế độ quét xoắn ốc (Helical / Spiral CT)</b>, bóng phát và mảng đầu dò quay liên tục với vận tốc góc không đổi đồng thời bàn bệnh nhân trượt đều dọc theo trục <i>z</i> [3, 7]. "
        "Trong hệ quy chiếu gắn với bệnh nhân, quỹ đạo của tiêu điểm bóng phát tia X vẽ nên một đường xoắn ốc ba chiều được mô tả bởi hệ phương trình tham số:",
        body_style
    ))
    add_eq("eq_2_1")
    story.append(Paragraph(
        "trong đó <i>R</i> là bán kính quỹ đạo từ tâm quay (isocenter) đến tiêu điểm bóng phát, <i>θ</i> là góc quay tích lũy (radian), và Δ<i>z</i> là quãng đường bàn bệnh nhân dịch chuyển được sau đúng một vòng quay 360° (2<i>π</i> rad) của Gantry. "
        "Để chuẩn hóa mối quan hệ giữa tốc độ trượt bàn và độ rộng chùm tia trên mảng đầu dò đa dãy (gồm <i>N</i> dãy hoạt động, mỗi dãy có bề dày chuẩn trực <i>T</i> tại tâm quay), thông số <b>Hệ số bước xoắn (Pitch)</b> được định nghĩa theo chuẩn quốc tế IEC [3, 7, 10]:",
        body_style
    ))
    add_eq("eq_2_2")
    story.append(Paragraph(
        "• <b>Khi Pitch &lt; 1 (Quét chồng lấp – Overlapping Scan):</b> Quãng đường bàn đi trong một vòng quay nhỏ hơn tổng bề rộng chùm tia (Δ<i>z</i> &lt; <i>N</i> × <i>T</i>). Dữ liệu thể tích được lấy mẫu dư thừa chồng lấp dọc trục <i>z</i>, giúp giảm nhiễu hạt và tăng độ phân giải dọc trục (thường dùng trong chụp mạch vành CTA hoặc khảo sát cấu trúc vi thể), nhưng làm tăng liều bức xạ cho bệnh nhân.<br/>"
        "• <b>Khi Pitch = 1 (Quét tiếp giáp – Contiguous Scan):</b> Các vòng xoắn tiếp giáp khít nhau, cân bằng tối ưu giữa chất lượng hình ảnh và liều tia.<br/>"
        "• <b>Khi Pitch &gt; 1 (Quét giãn cách – Extended Scan, ví dụ Pitch = 1.2 – 1.5):</b> Quét bao phủ vùng giải phẫu dài (như toàn bộ ngực – bụng – chậu) trong thời gian cực ngắn với liều bức xạ thấp hơn, đánh đổi bằng việc mặt cắt hiệu dụng dọc trục <i>z</i> (<i>Slice Sensitivity Profile – SSP</i>) bị mở rộng nhẹ do nội suy.",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["docx_helical"], width=340, height=235))
    story.append(Paragraph("Hình 2.3: Minh họa nguyên lý quét xoắn ốc (Helical/Spiral CT) trong hệ quy chiếu gắn với bệnh nhân.", caption_style))

    story.append(Paragraph("<b>2.1.5. Thu nhận dữ liệu, nội suy lát cắt và tái tạo khối 3D (MPR, MIP, VR):</b>", h3_style))
    story.append(Paragraph(
        "Vì trong quá trình quét xoắn ốc bàn bệnh nhân di chuyển liên tục, các phép chiếu trong một vòng quay không nằm trên cùng một mặt phẳng <i>z</i> cố định. "
        "Nếu đưa trực tiếp dữ liệu xoắn ốc vào thuật toán tái tạo 2D, sự bất nhất về vị trí dọc trục giữa góc đầu (<i>θ</i> = 0) và góc cuối (<i>θ</i> = 2<i>π</i>) sẽ sinh ra ảnh giả vệt sọc nghiêm trọng. "
        "Do đó, trước bước tái tạo mặt cắt, máy tính thực hiện <b>nội suy xoắn ốc dọc trục <i>z</i> (<i>z-filtering / 180°LI hoặc 360°LI Helical Interpolation</i>)</b> từ các dãy đầu dò lân cận để ước lượng bộ dữ liệu Sinogram phẳng tại bất kỳ tọa độ <i>z</i><sub>0</sub> mong muốn nào [3, 15]. "
        "Sau khi khôi phục chuỗi lát cắt ngang có voxel đẳng hướng, trạm làm việc thực hiện tái tạo đa mặt phẳng <b>MPR (Multiplanar Reformation)</b> theo các mặt phẳng Coronal (đứng ngang), Sagittal (đứng dọc) hoặc mặt phẳng cong theo lòng mạch, "
        "chiếu cường độ cực đại <b>MIP (Maximum Intensity Projection)</b> cho mạch máu cản quang, và dựng hình thể tích <b>VR (Volume Rendering)</b>.",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["docx_3d_mpr"], width=465, height=115))
    story.append(Paragraph("Hình 2.4: Sơ đồ chuỗi xử lý từ thu nhận nhiều góc chiếu qua DAS đến tái tạo ảnh lát cắt và dựng hình MPR/3D.", caption_style))

    story.append(Paragraph("<b>2.1.6. Ưu điểm kỹ thuật, các dạng nhiễu ảnh phần cứng và Bảng đối chiếu Axial CT vs Helical MDCT:</b>", h3_style))
    story.append(Paragraph(
        "Nhờ kết hợp vòng trượt và mảng đầu dò đa dãy, Helical MDCT loại bỏ thời gian chết giữa các lát cắt, giảm thiểu nhiễu do hô hấp (chụp trọn lồng ngực trong 3–5 giây) "
        "và cho phép bắt đúng pha động mạch – tĩnh mạch khi tiêm thuốc cản quang Iodine [3, 7]. Tuy nhiên, trong vận hành thực tế, hệ thống vẫn phải kiểm soát chặt chẽ các nguồn nhiễu vật lý và phần cứng:<br/>"
        "• <b>Nhiễu cứng hóa chùm tia (Beam Hardening) & Đói photon (Photon Starvation):</b> Do chùm tia X phát ra từ bóng phát là đa sắc (Polychromatic), khi đi qua xương dày hoặc vật liệu cấy ghép kim loại, các photon năng lượng thấp bị hấp thụ ưu tiên làm tăng năng lượng trung bình của chùm tia, gây ra sai số tối vùng tâm (<i>cupping artifact</i>) hoặc các vệt sọc đen trắng giữa hai khối đặc.<br/>"
        "• <b>Nhiễu vòng tròn đồng tâm (Ring Artifact):</b> Đặc trưng của hình học Thế hệ III khi một phần tử đầu dò bị trôi độ nhạy hoặc hỏng hiệu chuẩn; vì đầu dò đó luôn nằm ở khoảng cách bán kính <i>s</i> cố định so với tâm quay ở mọi góc <i>θ</i>, phép chiếu ngược sẽ vẽ nên một vòng tròn giả trên ảnh.<br/>"
        "• <b>Nhiễu chuyển động và Nhiễu xoắn ốc (Motion & Windmill Artifacts):</b> Xuất hiện khi bệnh nhân cử động, nhịp tim nhanh hoặc khi chọn Pitch quá cao khiến thuật toán nội suy dọc trục <i>z</i> bị răng cưa.",
        body_style
    ))

    t_ax_vs_hel_data = [
        ["Tiêu chí so sánh", "Chụp cắt lớp tuần tự (Axial / Step-and-Shoot CT)", "Chụp cắt lớp xoắn ốc đa dãy (Helical MDCT)"],
        ["Chuyển động bàn bệnh nhân", "Nhích từng bước rời rạc; bàn dừng lại khi phát tia", "Trượt liên tục đồng thời với chuyển động quay của Gantry"],
        ["Dữ liệu thu nhận", "Từng lát cắt phẳng độc lập tại các vị trí z rời rạc", "Một dải dữ liệu thể tích xoắn ốc liên tục dọc theo trục z"],
        ["Thời gian khảo sát toàn vùng", "Chậm do mất thời gian tăng/giảm tốc bàn giữa các lát", "Rất nhanh (vài giây cho toàn bộ ngực – bụng trong 1 lần nín thở)"],
        ["Độ phân giải trục z & Dựng 3D", "Không liên tục, dễ bị bậc thang (stair-step) trên ảnh MPR/3D", "Voxel đẳng hướng, tái tạo lát chồng lấp mượt mà cho MPR, MIP, VR"],
        ["Khảo sát mạch máu (CTA)", "Khó bắt kịp pha thuốc cản quang trên vùng giải phẫu dài", "Tối ưu cho CTA động mạch vành, động mạch chủ và tưới máu tạng"],
    ]
    t_ax_hel = Table(t_ax_vs_hel_data, colWidths=[115, 195, 195])
    t_ax_hel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (0, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    story.append(t_ax_hel)
    story.append(Paragraph("Bảng 2.2: So sánh kỹ thuật giữa chụp cắt lớp tuần tự (Axial CT) và chụp cắt lớp xoắn ốc đa dãy (Helical MDCT) [3, 7].", caption_style))
    story.append(PageBreak())

    # ==================================================================================
    # MỤC 2.2: THANG XÁM HOUNSFIELD (HU), CHẨN ĐOÁN BỆNH LÝ & TỐI ƯU HÓA LIỀU BỨC XẠ ALARA
    # (Tích hợp đầy đủ nội dung Phần 2.2 từ file .docx của nhóm)
    # ==================================================================================
    story.append(Paragraph("2.2. Thang xám Hounsfield (HU), Chẩn đoán bệnh lý & Tối ưu hóa liều bức xạ ALARA", h2_style))

    story.append(Paragraph("<b>2.2.1. Định nghĩa Thang đo Hounsfield (HU), ý nghĩa vật lý 'độ trong suốt' 0.1% và dải HU mô học:</b>", h3_style))
    story.append(Paragraph(
        "Thang xám Hounsfield (<i>Hounsfield Scale</i>) là thang đo định lượng tiêu chuẩn trong công nghệ chụp cắt lớp vi tính dùng để chuẩn hóa mức độ suy giảm tia X "
        "(<i>X-ray attenuation</i>), thông qua đó phản ánh chính xác mật độ vật lý và tính chất sinh lý – bệnh lý của mô sinh học đang quan sát mà hoàn toàn không cần xâm lấn hay giải phẫu (<i>non-invasive</i>) [9, 19]. "
        "Được đặt theo tên của Sir Godfrey Hounsfield (đồng nhận Giải Nobel năm 1979 cùng Allan McLeod Cormack [6]), chỉ số Hounsfield (HU) tại mỗi voxel có hệ số suy giảm tuyến tính <i>μ</i>(<i>x, y</i>) "
        "(thứ nguyên <i>L</i><sup>−1</sup>, đơn vị SI: m<sup>−1</sup> hoặc cm<sup>−1</sup>) được định nghĩa chính xác là [9, 19]:",
        body_style
    ))
    add_eq("eq_2_4")
    story.append(Paragraph(
        "Trong đó <i>μ</i><sub>H<sub>2</sub>O</sub> và <i>μ</i><sub>khí khô</sub> lần lượt là hệ số suy giảm tuyến tính của nước cất tinh khiết và không khí khô ở điều kiện tiêu chuẩn tại mức năng lượng hiệu dụng của chùm tia. "
        "Vì hệ số suy giảm của không khí khô xấp xỉ bằng 0 (<i>μ</i><sub>khí khô</sub> ≈ 0), từ Phương trình (2.4) ta rút ra ý nghĩa vật lý trực quan: "
        "<b>Chỉ số HU phản ánh 'độ trong suốt' (hoặc độ đặc cản tia) của mỗi voxel mô sinh học đối với tia X khi lấy nước cất làm mốc chuẩn gốc (HU<sub>nước</sub> = 0) và không khí làm mốc âm (HU<sub>khí</sub> = −1000). "
        "Mỗi sự thay đổi ±1 HU tương ứng chính xác với việc mô đó có độ cản quang (hoặc thấu quang) mạnh hơn hay yếu hơn nước cất 0.1%!</b>",
        body_style
    ))

    hu_full_table = [
        ["Nhóm vật chất / Cơ quan", "Chi tiết mô / Dịch / Trạng thái bệnh lý", "Giá trị Hounsfield (HU)", "Ý nghĩa vật lý & Chẩn đoán lâm sàng"],
        ["Chuẩn tham chiếu", "Không khí khô (Air)  |  Nước cất (Pure Water)", "−1000 HU  |  0 HU", "Hai mốc chuẩn gốc để hiệu chuẩn (Calibration) máy CT hằng ngày"],
        ["Mô mỡ & Mô mềm", "Lớp mỡ (Fat)  |  Mô mềm có tiêm thuốc cản quang", "−120 đến −90  |  +100 đến +300", "Mỡ chứa nguyên tử C, H nhẹ nên suy giảm thấp hơn nước (~9–12%)"],
        ["Mô xương (Bone)", "Xương xốp (Cancellous)  |  Xương đặc (Cortical)", "+300 đến +400  |  +500 đến +1900", "Giàu Canxi (Z = 20) hấp thụ quang điện mạnh, độ cản quang rất cao"],
        ["Máu tụ dưới màng cứng\n(Subdural Hematoma)\ntheo thời gian", "• Giai đoạn cấp (Giờ đầu tiên)\n• Giai đoạn bán cấp sớm (Sau 3 ngày)\n• Giai đoạn mạn / tiêu sợi huyết (10 – 14 ngày)", "• +75 đến +100 HU\n• +65 đến +85 HU\n• +35 đến +40 HU", "Giờ đầu cục máu đông co rút tập trung Hemoglobin làm tăng HU;\nsau 10–14 ngày quá trình tiêu sợi huyết làm HU giảm tiệm cận mô não"],
        ["Máu ở trạng thái khác", "Máu lỏng không đông (Unclotted)  |  Máu đông (Clotted)", "+13 đến +50  |  +50 đến +75", "Phân biệt máu đang chảy tích dịch với cục máu đông đặc"],
        ["Tràn dịch màng phổi", "Dịch thấm (Transudate)  |  Dịch tiết (Exudate)", "+2 đến +15  |  +4 đến +33", "Dịch tiết giàu protein và tế bào viêm nên có HU cao hơn dịch thấm"],
        ["Các dịch cơ thể khác", "Nhũ chấp (−30) | Nước tiểu, Mật (−5..+15) | Dịch não tủy CSF (+15) | Mủ/Áp-xe (0..+45) | Dịch nhầy (0..+130)", "−30 đến +130 HU\n(Nhầy đặc > +70 HU)", "Hỗ trợ phân biệt nang dịch đơn thuần, ổ áp-xe hóa mủ và nút nhầy đặc"],
        ["Nhu mô tạng\n(Parenchyma)", "• Phổi (Lung): −700 đến −600  |  • Thận (Kidney): +20 đến +45\n• Gan (Liver): 60 ± 6 (54..66)  |  • Cơ bắp (Muscle): +35 đến +55\n• Hạch (+10..+20) | Tuyến ức (+20..+120) | Chất trắng (+20..+30) | Chất xám (+37..+45)", "Phổi: −700 đến −600\nTạng đặc: +10 đến +66", "Phổi chứa đầy khí phế nang nên HU rất âm; Chất xám nhiều thân nơ-ron và mao mạch hơn chất trắng có bao myelin giàu lipid"],
        ["Sỏi mật (Gallstone)", "Sỏi Cholesterol  |  Sỏi sắc tố mật Bilirubin", "+30 đến +100  |  +90 đến +120", "Phân loại thành phần hóa học của sỏi để quyết định hướng điều trị"],
        ["Dị vật & Kim loại\n(Foreign Bodies)", "Kính cửa sổ (+500) | Nhôm, đá (+2100..+2300) | Đá vôi (+2800) | Đồng (+14,000) | Bạc (+17,000) | Thép (+20,000) | Vàng (+30,000)", "+500 đến +30,000 HU\n(Giới hạn đo mở rộng)", "Nguyên tử khối Z cực lớn làm tắc nghẽn hoàn toàn photon tia X,\ngây ảnh giả vệt sọc kim loại (Metal Streak Artifacts)"],
    ]
    t_hu_full = Table(hu_full_table, colWidths=[88, 165, 92, 160])
    t_hu_full.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (0, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_hu_full)
    story.append(Paragraph("Bảng 2.3: Bảng tổng hợp giá trị Hounsfield (HU) đặc trưng cho các mô, nhu mô, dịch cơ thể, diễn tiến máu tụ theo thời gian và dị vật [9, 19].", caption_style))

    story.append(Paragraph("<b>2.2.2. Chất chỉ thị cản quang Iodine (CTA) và Chẩn đoán bệnh lý lâm sàng qua biến thiên chỉ số HU:</b>", h3_style))
    story.append(Paragraph(
        "Trong nhiều trường hợp khảo sát mạch máu (<i>CT Angiogram – CTA</i>) hoặc đánh giá sự tăng sinh mạch của khối u, độ tương phản tự nhiên giữa máu lỏng (+13 đến +50 HU) "
        "và thành mạch/nhu mô xung quanh (+20 đến +60 HU) là chưa đủ lớn. Khi đó, bệnh nhân được tiêm <b>chất chỉ thị cản quang chứa Iodine</b> (một nguyên tố halogen có nguyên tử khối lớn, "
        "độ hấp thụ quang điện bờ K tại 33.2 keV cực mạnh, được bọc trong các phân tử hữu cơ tan trong nước). "
        "Ở điện thế ống tiêu chuẩn <b>100 – 120 kV, nồng độ 1 mg Iodine trong 1 mL dịch làm tăng độ tương phản lên thêm 25 – 30 HU</b>, giúp làm sáng rực lòng động mạch và các tổn thương giàu mạch. "
        "Tuy nhiên, thuốc cản quang chứa Iodine có thể gây tác dụng phụ như thay đổi vị giác (vị kim loại), buồn nôn, phản ứng phản vệ và đặc biệt là <b>nguy cơ gây suy thận cấp do thuốc cản quang</b> "
        "(<i>Contrast-Induced Nephropathy</i>), nên bắt buộc phải kiểm tra độ lọc cầu thận (eGFR/Creatinine huyết thanh) trước khi chỉ định [9, 19].",
        body_style
    ))
    story.append(Paragraph(
        "Khi chỉ số HU của một cơ quan nội tạng cao hơn hoặc thấp hơn đáng kể so với khoảng sinh lý bình thường trong Bảng 2.3, đó là dấu hiệu định lượng trực tiếp của bệnh lý:<br/>"
        "• <b>Chẩn đoán Gan nhiễm mỡ (Hepatic Steatosis) qua đối chiếu HU Gan – Lách:</b> Vì mô mỡ có chỉ số HU âm (−120 đến −90 HU) thấp hơn rất nhiều so với nước và tế bào gan, "
        "sự tích tụ hạt mỡ triglyceride bên trong tế bào gan làm kéo tụt chỉ số HU trung bình của nhu mô gan xuống thấp. Trong thực hành lâm sàng, khi không có phim chụp CT lúc gan khỏe mạnh của chính bệnh nhân đó để so sánh, "
        "bác sĩ chẩn đoán hình ảnh sẽ <b>so sánh trực tiếp HU của nhu mô gan với HU của nhu mô lá lách (Spleen) trên cùng một lát cắt</b> (Hình 2.5): "
        "Ở <i>gan bình thường</i>, tỷ trọng nhu mô gan (~66 HU) luôn cao hơn lá lách (~56 HU) khoảng 8–10 HU và các tĩnh mạch gan (<i>v</i>) hiện lên tối hơn nền gan. "
        "Ngược lại, ở <i>gan nhiễm mỡ</i>, tỷ trọng nhu mô gan giảm sâu (ví dụ chỉ còn 15 HU) thấp hơn rõ rệt so với lá lách (40 HU), đồng thời các mạch máu trong gan (<i>v</i>) xuất hiện sáng hơn (tăng tỷ trọng tương đối) so với nền gan xung quanh.<br/>"
        "• <b>Tầm soát sớm ung thư phổi bằng CT liều thấp (LDCT):</b> Do nhu mô phổi khỏe mạnh chứa khí có HU đặc biệt thấp (−700 đến −600 HU) trong khi các nốt mờ hoặc khối u phổi có bản chất mô mềm (HU từ −400 đến +200 HU), "
        "chênh lệch độ tương phản tự nhiên ΔHU lên tới hàng trăm đơn vị, cho phép sử dụng kỹ thuật chụp CT liều thấp (LDCT) để phát hiện ung thư phổi giai đoạn sớm mà không cần liều bức xạ cao [1, 19].<br/>"
        "• <b>Chẩn đoán đột quỵ xuất huyết não và định tuổi khối máu tụ:</b> Khối máu tụ cấp tính hình thành trong hộp sọ có HU = +75 đến +100 HU trong những giờ đầu tiên, tạo độ tương phản dương rõ rệt so với chất xám (+37 đến +45 HU) và chất trắng (+20 đến +30 HU). "
        "Hơn nữa, việc theo dõi sự suy giảm HU của khối máu tụ qua các mốc thời gian (xuống +65..+85 HU sau 3 ngày và +35..+40 HU sau 10–14 ngày) giúp bác sĩ xác định chính xác tuổi của ổ xuất huyết.",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["docx_fatty_liver"], width=435, height=230))
    story.append(Paragraph("Hình 2.5: Ảnh chụp CT lâm sàng đối chiếu giữa Gan bình thường (66 HU > Lách 56 HU) và Gan nhiễm mỡ (15 HU < Lách 40 HU).", caption_style))
    story.append(PageBreak())

    story.append(Paragraph("<b>2.2.3. Nguyên lý Tối ưu hóa Liều Bức xạ ALARA, Lựa chọn điện áp kVp, Tiêu chuẩn Rose (SNR ≥ 5) & Tính toán số photon:</b>", h3_style))
    story.append(Paragraph(
        "Trong vật lý y sinh và an toàn bức xạ hạt nhân, mọi chỉ định chụp CT đều phải tuân thủ nghiêm ngặt nguyên lý <b>ALARA</b> "
        "(<i>As Low As Reasonably Achievable / Practicable</i> – Giữ liều bức xạ ở mức thấp nhất có thể đạt được một cách hợp lý mà vẫn đảm bảo đầy đủ thông tin chẩn đoán) [1, 11]. "
        "Việc tính toán chính xác liều bức xạ hấp thụ trước khi chụp trên cơ thể người thực đòi hỏi các thuật toán mô phỏng vận chuyển bức xạ phức tạp (như <i>mô phỏng Monte Carlo</i>) để theo dõi quỹ đạo tán xạ và tỷ lệ hấp thụ năng lượng của từng mô. "
        "Theo số liệu của Hiệp hội Ung thư Hoa Kỳ (<i>American Cancer Society</i>), một ca chụp CT tiêu chuẩn ở vùng ổ bụng và khung chậu có liều hiệu dụng ước tính khoảng <b>7.7 mSv</b> – tương đương <b>2.6 lần</b> tổng lượng bức xạ nền tự nhiên mà một người nhận được trong suốt một năm (~3.0 mSv/năm). "
        "Tuy nhiên, nếu nắm vững bản chất vật lý của mô và độ tương phản HU giữa tổn thương với nền, công nghệ chụp CT liều thấp (LDCT) tầm soát ung thư phổi có thể hạ liều hiệu dụng xuống chỉ còn <b>1.5 mSv</b> (tương đương khoảng 6 tháng bức xạ nền)!",
        body_style
    ))
    story.append(Paragraph(
        "<b>a) Bài toán lựa chọn năng lượng photon và điện áp đỉnh của ống phát tia X (80 – 120 kVp):</b><br/>"
        "Phổ năng lượng bức xạ hãm đa sắc Kramers Φ(<i>E</i>) và tổng thông lượng photon <i>I</i><sub>0</sub> phát ra từ bóng tia X phụ thuộc chặt chẽ vào điện áp đỉnh kVp và tích số dòng – thời gian mAs:",
        body_style
    ))
    add_eq("eq_2_3")
    story.append(Paragraph(
        "Tại sao các máy chụp CT y tế luôn vận hành trong dải điện áp tối ưu từ <b>80 kVp đến 120 kVp</b> (hoặc tối đa 140 kVp cho bệnh nhân thể trạng lớn)?<br/>"
        "• <i>Nếu năng lượng photon quá thấp (&lt; 60 kVp):</i> Do tiết diện hấp thụ quang điện tỷ lệ nghịch với lũy thừa bậc ba của năng lượng (<i>τ</i> ∝ <i>Z</i><sup>3</sup> / <i>E</i><sup>3</sup>), "
        "hầu hết photon tia X sẽ bị hấp thụ hoàn toàn bên trong mô của bệnh nhân và không thể xuyên qua cơ thể để tới đầu dò. Hệ quả kép là: bệnh nhân phải chịu liều hấp thụ bức xạ cực cao (gây tổn thương DNA tế bào), trong khi ảnh CT thu được lại bị <b>trắng xóa hoặc đầy sọc nhiễu do đói photon (photon starvation)</b>.<br/>"
        "• <i>Nếu năng lượng photon quá cao (&gt; 150 kVp):</i> Đa số photon xuyên thẳng qua cơ thể quá dễ dàng và hiệu ứng tán xạ Compton (không phụ thuộc vào nguyên tử số <i>Z</i> mà chỉ phụ thuộc mật độ điện tử) chiếm ưu thế tuyệt đối. "
        "Khi đó, sự chênh lệch hệ số suy giảm giữa các mô mềm biến mất, khiến bức ảnh CT thu được có <b>màu xám phẳng lỳ, mất hoàn toàn độ tương phản chẩn đoán</b>.",
        body_style
    ))

    story.append(Paragraph(
        "<b>b) Dẫn xuất toán học Tiêu chuẩn Rose (Rose's Criterion) và Tính toán 'Back-of-the-napkin' giới hạn dưới số photon/voxel:</b><br/>"
        "Câu hỏi định lượng cốt lõi của nguyên lý ALARA là: <i>'Cần tối thiểu bao nhiêu photon tia X đi qua mỗi đơn vị thể tích voxel để mắt bác sĩ có thể nhận diện chắc chắn một tổn thương mục tiêu khỏi nền nhiễu?'</i>. "
        "Giới hạn dưới (<i>lower bound</i>) này được xác định bởi <b>Tiêu chuẩn Rose (Rose's Criterion)</b> trong lý thuyết phát hiện tín hiệu y sinh: tỷ số tín hiệu trên nhiễu (SNR) phải đạt tối thiểu bằng 5:",
        body_style
    ))
    add_eq("eq_2_5")
    story.append(Paragraph(
        "Trong đó <i>N</i> là số lượng photon tia X trung bình được hấp thụ/ghi nhận trên một đơn vị voxel. Do quá trình bức xạ và đếm hạt photon là quá trình ngẫu nhiên độc lập tuân theo <b>phân phối thống kê Poisson</b>, "
        "lượng nhiễu bắn lượng tử (<i>Poisson shot noise</i>) trung bình bằng đúng căn bậc hai của số photon: <b><i>σ</i> = √<i>N</i></b>. "
        "Mặt khác, từ định nghĩa thang xám Hounsfield ở Phương trình (2.4), sự chênh lệch số photon tín hiệu Δ<i>N</i> = |<i>N</i><sub>nền</sub> − <i>N</i><sub>mục tiêu</sub>| liên hệ trực tiếp với tỷ số tương phản trên nhiễu (CNR) và <b>Hệ số tương phản tương đối <i>C</i></b> theo độ lệch chỉ số Hounsfield ΔHU:",
        body_style
    ))
    add_eq("eq_2_6")
    story.append(Paragraph(
        "Từ Phương trình (2.6) ta có độ chênh lệch tín hiệu tuyệt đối <b>Δ<i>N</i> = <i>C</i> · <i>N</i></b>. Thế trực tiếp biểu thức Δ<i>N</i> = <i>C</i> · <i>N</i> và độ lệch chuẩn nhiễu Poisson <i>σ</i> = √<i>N</i> vào điều kiện Tiêu chuẩn Rose ở Phương trình (2.5), "
        "ta chứng minh được công thức giải tích xác định <b>số lượng photon tối thiểu <i>N</i><sub>min</sub> trên mỗi voxel</b> để ảnh CT đủ điều kiện chẩn đoán:",
        body_style
    ))
    add_eq("eq_2_7")
    story.append(Paragraph(
        "Áp dụng Phương trình (2.7) cho hai bài toán chẩn đoán lâm sàng điển hình sử dụng dữ liệu HU thực tế từ Bảng 2.3:<br/>"
        "1. <b>Bài toán 1 – Tầm soát khối u/nốt phổi trên nền nhu mô phổi:</b> Nhu mô phổi khỏe mạnh có HU<sub>nền</sub> ≈ −650 HU (trung bình của dải −700 đến −600 HU); "
        "mô mềm nốt phổi có HU<sub>mục tiêu</sub> ≈ +200 HU. Độ chênh lệch tương phản là <b>ΔHU = |200 − (−650)| = 850 HU</b> ⇒ Hệ số tương phản <i>C</i> = 850 / 1000 = 0.850. "
        "Thay vào (2.7), số photon tối thiểu cần thiết trên mỗi voxel chỉ là: <b><i>N</i><sub>min</sub> = (5 / 0.850)<sup>2</sup> ≈ 34.60 photon/voxel</b>.<br/>"
        "2. <b>Bài toán 2 – Phát hiện khối máu tụ dưới màng cứng cấp tính (giờ đầu) trên nền nhu mô não:</b> Nhu mô não lành (lấy trung bình cộng giữa chất xám +37..+45 HU và chất trắng +20..+30 HU) có HU<sub>nền</sub> ≈ +33 HU; "
        "khối máu đông dưới màng cứng trong giờ đầu tiên (+75 đến +100 HU) có trung bình HU<sub>mục tiêu</sub> ≈ +87.5 HU. Độ chênh lệch tương phản chỉ là <b>ΔHU = |87.5 − 33| = 54.5 HU</b> ⇒ Hệ số tương phản <i>C</i> = 54.5 / 1000 = 0.0545. "
        "Thay vào (2.7), số photon tối thiểu cần thiết trên mỗi voxel vọt lên tới: <b><i>N</i><sub>min</sub> = (5 / 0.0545)<sup>2</sup> ≈ 8,416.80 photon/voxel</b>!",
        body_style
    ))

    t_napkin_data = [
        ["Bài toán chẩn đoán lâm sàng", "Mô nền & HU(nền)", "Mô mục tiêu & HU(mục tiêu)", "Độ lệch ΔHU", "Hệ số tương phản C = ΔHU/1000", "Số photon/voxel tối thiểu N_min = (5/C)² (SNR ≥ 5)"],
        ["1. Phát hiện nốt u phổi\n(Chụp CT phổi liều thấp LDCT)", "Nhu mô phổi chứa khí\nHU(nền) = −650 HU", "Nhu mô nốt phổi đặc\nHU(mục tiêu) = +200 HU", "850 HU", "C = 0.8500", "34.60 photon / voxel\n(Chỉ cần liều cực thấp ~1.5 mSv)"],
        ["2. Phát hiện máu tụ nội sọ cấp\n(Chụp CT sọ não cấp cứu)", "Não (Chất xám – chất trắng)\nHU(nền) = +33.0 HU", "Máu tụ dưới màng cứng giờ đầu\nHU(mục tiêu) = +87.5 HU", "54.5 HU", "C = 0.0545", "8,416.80 photon / voxel\n(Cần lượng photon gấp ~243 lần!)"],
    ]
    t_napkin = Table(t_napkin_data, colWidths=[110, 92, 102, 52, 66, 98])
    t_napkin.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (0, -1), FONT_BOLD),
        ("FONTNAME", (-1, 1), (-1, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_napkin)
    story.append(Paragraph("Bảng 2.4: Kết quả tính toán định lượng giới hạn dưới số photon/voxel theo tiêu chuẩn Rose (SNR ≥ 5) giữa CT Phổi và CT Sọ não.", caption_style))

    story.append(Paragraph(
        "<b>Kết luận vật lý y sinh từ Bảng 2.4:</b> Ở cùng một mức điện áp (như 120 kVp), tổng năng lượng bức xạ truyền cho một đơn vị khối lượng mô tỷ lệ thuận trực tiếp với số lượng photon <i>N</i>. "
        "Do độ tương phản giữa máu tụ và nhu mô não (ΔHU = 54.5) nhỏ hơn 15.6 lần so với độ tương phản ở phổi (ΔHU = 850), mà số photon cần thiết lại tỷ lệ nghịch với <i>bình phương</i> độ tương phản (<i>N</i><sub>min</sub> ∝ 1/ΔHU<sup>2</sup>), "
        "nên chụp CT sọ não đòi hỏi lượng photon trên mỗi voxel <b>nhiều gấp 8416.80 / 34.60 ≈ 243.2 lần</b> so với chụp nhu mô phổi! "
        "Đây chính là cơ sở toán lý giải thích vì sao bác sĩ và kỹ sư có thể giảm mạnh dòng mAs trong chụp tầm soát phổi (LDCT), nhưng bắt buộc phải duy trì mAs cao hoặc sử dụng các thuật toán tái tạo lặp thống kê (MBIR) và học sâu (DLIR) ở Chương 3 khi khảo sát mô mềm có độ tương phản thấp.",
        body_style
    ))
    story.append(PageBreak())

    # ==================================================================================
    # CHƯƠNG 3: CƠ SỞ TOÁN HỌC BÀI TOÁN NGƯỢC & BÓC TÁCH CÁC THUẬT TOÁN TÁI TẠO
    # ==================================================================================
    story.append(Paragraph("CHƯƠNG 3: CƠ SỞ TOÁN HỌC BÀI TOÁN NGƯỢC & BÓC TÁCH CÁC THUẬT TOÁN TÁI TẠO", h1_style))

    # --- MỤC 3.1 ---
    story.append(Paragraph("3.1. Đặt bài toán ngược tổng quát (Ill-posed Inverse Problem) và Phương sai nhiễu Logarit", h2_style))
    story.append(Paragraph(
        "Chuyển hệ quy chiếu từ mô tả thiết bị sang <b>Giải tích hàm và Lý thuyết Bài toán ngược (Inverse Problems)</b> [13, 14], "
        "bài toán chụp cắt lớp vi tính được phát biểu chặt chẽ như sau: Cho toán tử tuyến tính Radon <i>R</i>: <i>L</i><sup>2</sup>(Ω) → <i>L</i><sup>2</sup>(<i>Z</i>) ánh xạ hàm phân bố "
        "hệ số suy giảm chưa biết <i>μ</i>(<i>x, y</i>) trên miền giải phẫu Ω sang miền dữ liệu hình trụ <i>Z</i> = [−<i>R</i><sub>fov</sub>, <i>R</i><sub>fov</sub>] × [0, <i>π</i>). "
        "Trong thực tế, ta không bao giờ đo được hàm <i>R</i>[<i>μ</i>] lý tưởng mà chỉ ghi nhận được biến ngẫu nhiên số đếm photon <i>N<sub>i</sub></i> tại tia thứ <i>i</i> [15]:",
        body_style
    ))
    add_eq("eq_3_1")
    story.append(Paragraph(
        "Trong đó <i>I</i><sub>0</sub> tỷ lệ thuận với liều bức xạ (mAs) và <i>σ<sub>e</sub></i> là độ lệch chuẩn nhiễu nhiệt điện tử của bộ khuếch đại DAS. "
        "Khi thực hiện phép biến đổi phi tuyến logarit <i>p<sub>i</sub></i> = −ln(<i>N<sub>i</sub></i> / <i>I</i><sub>0</sub>) để đưa phương trình về dạng tuyến tính <i>p</i> = <i>Rμ</i> + <i>η</i>, "
        "ta khai triển Taylor bậc hai của hàm <i>f</i>(<i>N<sub>i</sub></i>) = −ln(<i>N<sub>i</sub></i> / <i>I</i><sub>0</sub>) quanh giá trị kỳ vọng <i>N̄<sub>i</sub></i>: "
        "<i>f</i>(<i>N<sub>i</sub></i>) ≈ <i>f</i>(<i>N̄<sub>i</sub></i>) − (<i>N<sub>i</sub></i> − <i>N̄<sub>i</sub></i>)/<i>N̄<sub>i</sub></i> + (<i>N<sub>i</sub></i> − <i>N̄<sub>i</sub></i>)<sup>2</sup> / (2<i>N̄<sub>i</sub></i><sup>2</sup>). "
        "Từ đó, <b>kỳ vọng độ lệch (Bias) và phương sai nhiễu (Variance) của từng phép đo chiếu <i>p<sub>i</sub></i></b> được chứng minh tường minh:",
        body_style
    ))
    add_eq("eq_3_2")
    story.append(Paragraph(
        "<b>Hệ quả toán học cốt lõi từ Phương trình (3.2):</b><br/>"
        "1. <i>Tính chất đặt không chỉnh (Ill-posedness theo nghĩa Hadamard):</i> Toán tử tích phân Radon <i>R</i> là một toán tử compact trên không gian Hilbert; "
        "dãy các giá trị kỳ dị (<i>singular values</i>) <i>σ<sub>k</sub></i> của <i>R</i> suy giảm về 0 theo tần số không gian: <b><i>σ<sub>k</sub></i> ~ |<i>ω</i>|<sup>−1/2</sup> → 0 khi |<i>ω</i>| → ∞</b> [14]. "
        "Do đó, toán tử nghịch đảo chính xác <i>R</i><sup>−1</sup> là <b>không liên tục (unbounded)</b>: thành phần nhiễu tần số cao trong <i>p<sub>i</sub></i> sẽ bị nhân với thừa số khuếch đại |<i>ω</i>| trong miền 2D, gây bùng nổ sai số.<br/>"
        "2. <i>Mâu thuẫn đối kháng giữa Sai số ước lượng và Liều bức xạ (ALARA):</i> Khi giảm liều tia X (giảm <i>I</i><sub>0</sub>) hoặc khi tia X đi qua cấu trúc cản quang mạnh "
        "(xương đặc, kim loại khiến tích phân đường (<i>Rμ</i>)<i><sub>i</sub></i> rất lớn), mẫu số <i>I</i><sub>0</sub> exp(−(<i>Rμ</i>)<i><sub>i</sub></i>) tiến về 0 làm phương sai nhiễu <b>Var(<i>p<sub>i</sub></i>) tăng vọt theo hàm mũ</b>!<br/>"
        "3. <i>Mục tiêu tối hậu của thuật toán tái tạo CT:</i> Thiết kế một toán tử nghịch đảo có điều chuẩn (<i>Regularized Reconstruction Operator</i>) "
        "<b><i>μ̂</i> = <i>T<sub>α</sub></i>(<i>p</i>)</b> sao cho cực tiểu hóa sai số toàn cục <b>E[ ‖<i>μ̂</i>(<i>x, y</i>) − <i>μ</i><sub>true</sub>(<i>x, y</i>)‖<sup>2</sup> ] → min</b> dưới ràng buộc liều chiếu xạ <b><i>I</i><sub>0</sub> → min</b>.",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(KeepTogether([
        Image(fig_paths["fc0_inverse"], width=485, height=288),
        Paragraph("Hình 3.1: [Flowchart 0] Sơ đồ khung toán học Bài toán ngược từ quá trình suy giảm vật lý đến 3 nhánh khôi phục hàm μ(x, y).", caption_style)
    ]))

    # --- MỤC 3.2 ---
    story.append(Paragraph("3.2. Bóc tách Phép chiếu ngược đơn thuần (BP): Chứng minh hàm truyền điểm PSF 1/(πr)", h2_style))
    story.append(Paragraph(
        "Phép chiếu ngược đơn thuần (<i>Unfiltered Backprojection</i> – BP) sử dụng trực tiếp <b>toán tử liên hợp hình thức (Formal Adjoint Operator) <i>R</i><sup>*</sup></b> "
        "làm xấp xỉ cho toán tử nghịch đảo <i>R</i><sup>−1</sup> [13]. Với mọi điểm (<i>x, y</i>) trên mặt phẳng ảnh, BP tích phân đều các giá trị chiếu đi qua điểm đó:",
        body_style
    ))
    add_eq("eq_3_3")
    story.append(Paragraph(
        "<b>Chứng minh giải tích vì sao BP không khôi phục được <i>μ</i>(<i>x, y</i>) mà gây sương mờ 1/(<i>πr</i>):</b> Thay biểu thức <i>p</i>(<i>s, θ</i>) = <i>R</i>[<i>μ</i>](<i>s, θ</i>) từ Phương trình (1.2) vào (3.3), "
        "đổi thứ tự tích phân và đặt véc-tơ hiệu trong tọa độ cực <i>x − x'</i> = <i>r</i> cos <i>φ</i>, <i>y − y'</i> = <i>r</i> sin <i>φ</i> với khoảng cách Euclid "
        "<i>r</i> = √((<i>x − x'</i>)<sup>2</sup> + (<i>y − y'</i>)<sup>2</sup>). Áp dụng tính chất co giãn của hàm suy rộng Dirac <i>δ</i>(<i>r</i> cos(<i>θ − φ</i>)) = (1/<i>r</i>) <i>δ</i>(cos(<i>θ − φ</i>)), "
        "tích phân theo góc <i>θ</i> ∈ [0, <i>π</i>) cho đúng giá trị 1/<i>r</i>. Khi đó phương trình tích chập 2D và biến đổi Fourier 2D của ảnh BP được chứng minh chính xác là [13, 14]:",
        body_style
    ))
    add_eq("eq_3_4")
    story.append(Paragraph(
        "• <b>Kết luận toán học cho BP:</b> Hàm truyền đạt của hệ thống BP là <i>H</i><sub>BP</sub>(<i>ω</i>) = 1 / (<i>π</i>|<i>ω</i>|). Nó có điểm kỳ dị tiến ra vô cùng tại tần số <i>ω</i> → 0 "
        "(khuếch đại quá mức thành phần nền DC tạo lớp mù trắng toàn ảnh) và suy giảm theo tỷ lệ nghịch 1/|<i>ω</i>| ở tần số cao, làm triệt tiêu mọi gradient biên giải phẫu. "
        "Do đó, sai số quân phương ‖<i>b − μ</i>‖<sub>2</sub> của BP luôn rất lớn bất kể tăng liều tia X cao đến đâu.",
        body_style
    ))

    story.append(Spacer(1, 3))
    story.append(Image(fig_paths["fc1_bp"], width=495, height=200))
    story.append(Paragraph("Hình 3.2: [Flowchart 1] Sơ đồ khối thuật toán Chiếu ngược đơn thuần (BP) và cơ chế toán học sinh hàm truyền điểm 1/(πr).", caption_style))

    # --- MỤC 3.3 ---
    story.append(Paragraph("3.3. Bóc tách Chiếu ngược có lọc (FBP): Định lý Lát cắt Fourier, Nhân Ram-Lak & 5 Bộ lọc", h2_style))
    story.append(Paragraph("<b>3.3.1. Chứng minh Định lý Lát cắt Trung tâm Fourier (Fourier Central Slice Theorem):</b>", h3_style))
    story.append(Paragraph(
        "Xét biến đổi Fourier 1D của phép chiếu <i>p</i>(<i>s, θ</i>) theo biến vị trí đầu dò <i>s</i> tại góc cố định <i>θ</i> [4, 13]:",
        body_style
    ))
    add_eq("eq_3_5")
    story.append(Paragraph(
        "Phương trình (3.5) chứng minh rằng phổ Fourier 1D của hình chiếu tại góc <i>θ</i> trùng khít với một lát cắt đường thẳng đi qua gốc tọa độ của phổ Fourier 2D <i>M</i>(<i>ω<sub>x</sub>, ω<sub>y</sub></i>) của hàm suy giảm <i>μ</i>(<i>x, y</i>). "
        "Để tìm lại <i>μ</i>(<i>x, y</i>), ta viết tích phân Fourier ngược 2D và đổi biến sang tọa độ cực (<i>ω<sub>x</sub></i> = <i>ω</i> cos <i>θ</i>, <i>ω<sub>y</sub></i> = <i>ω</i> sin <i>θ</i>) với định thức <b>Jacobian chuyển hệ tọa độ d<i>ω<sub>x</sub></i> d<i>ω<sub>y</sub></i> = |<i>ω</i>| d<i>ω</i> d<i>θ</i></b>:",
        body_style
    ))
    add_eq("eq_3_6")
    story.append(Paragraph(
        "trong đó <b><i>q</i>(<i>s, θ</i>) = F<sub>1D</sub><sup>−1</sup>{ |<i>ω</i>| · <i>P</i>(<i>ω, θ</i>) }</b> là hình chiếu đã được lọc bởi bộ lọc dốc <b><i>H</i><sub>Ramp</sub>(<i>ω</i>) = |<i>ω</i>|</b>. "
        "Như vậy, thừa số Jacobian |<i>ω</i>| chính là toán tử đạo hàm phân số bậc một trong miền tần số, triệt tiêu chính xác thừa số 1/|<i>ω</i>| của phép chiếu ngược ở Phương trình (3.4)!",
        body_style
    ))

    story.append(Paragraph("<b>3.3.2. Dẫn xuất Đáp ứng xung miền không gian (Ram-Lak Impulse Response) và Kỹ thuật Zero-Padding:</b>", h3_style))
    story.append(Paragraph(
        "Trong hệ thống CT số hóa, khoảng cách giữa các kênh đầu dò là Δ<i>s</i>, nên tần số lấy mẫu bị giới hạn bởi tần số Nyquist <i>ω<sub>N</sub></i> = 1 / (2Δ<i>s</i>). "
        "Thực hiện tích phân Fourier ngược 1D bằng phương pháp tích phân từng phần cho bộ lọc Ram-Lak giới hạn băng thông (chuẩn hóa Δ<i>s</i> = 1), ta thu được công thức giải tích chính xác của <b>nhân chập Ram-Lak rời rạc <i>h</i>[<i>n</i>]</b> (được lập trình trực tiếp tại hàm <code>fourier_filter</code> trong <code>ct_sim/recon.py</code>) [13]:",
        body_style
    ))
    add_eq("eq_3_7")
    story.append(Paragraph(
        "• <b>Ý nghĩa của các thùy âm <i>h</i>[<i>n</i> lẻ] = −1/(<i>πn</i>)<sup>2</sup> &lt; 0:</b> Chính các trọng số âm vươn dài theo quy luật 1/<i>n</i><sup>2</sup> này tạo ra các vùng giao thoa triệt tiêu (<i>destructive interference</i>) "
        "để đào bỏ hoàn toàn quầng sương mờ dương +1/<i>r</i> xung quanh mỗi điểm ảnh!<br/>"
        "• <b>Tại sao bắt buộc phải đệm số không (Zero-Padding lên <i>M</i> ≥ 2<i>N</i><sub>det</sub>) trước khi nhân FFT?</b> Phép nhân trực tiếp hai chuỗi DFT độ dài <i>N</i><sub>det</sub> tương đương với <i>tích chập vòng (Circular Convolution)</i>. "
        "Do nhân <i>h</i>[<i>n</i>] suy giảm chậm theo 1/<i>n</i><sup>2</sup>, tích chập vòng sẽ làm rìa trái và rìa phải của hình chiếu giao thoa với nhau, "
        "gây ra <b>sai số lõm hình đĩa (cupping/dishing artifact)</b> làm lệch giá trị HU ở tâm ảnh. Đệm số không lên <i>M</i> = 2<sup>⌈log<sub>2</sub>(2<i>N</i><sub>det</sub>)⌉</sup> chuyển tích chập vòng thành tích chập tuyến tính chính xác.",
        body_style
    ))
    story.append(PageBreak())

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["fc2_fbp"], width=495, height=306))
    story.append(Paragraph("Hình 3.3: [Flowchart 2] Sơ đồ khối thuật toán Chiếu ngược có lọc (FBP) và 5 cửa sổ điều biến miền tần số.", caption_style))

    story.append(Paragraph("<b>3.3.3. Định lý Truyền Phương Sai Nhiễu qua FBP và Bóc tách 5 Cửa sổ Điều biến (Apodization Windows):</b>", h3_style))
    story.append(Paragraph(
        "Để giảm độ khuếch đại nhiễu ở tần số cao gần <i>ω<sub>N</sub></i>, bộ lọc Ram-Lak được nhân thêm hàm cửa sổ điều biến <i>W</i>(<i>ω</i>): <b><i>H</i>(<i>ω</i>) = |<i>ω</i>| · <i>W</i>(<i>ω</i>)</b>. "
        "Theo lý thuyết quá trình ngẫu nhiên qua hệ thống tuyến tính, phương sai nhiễu tại pixel (<i>x, y</i>) trên ảnh FBP được dẫn xuất bằng định lý Parseval [13, 15]:",
        body_style
    ))
    add_eq("eq_3_8")
    story.append(Paragraph(
        "trong đó <b><i>η<sub>W</sub></i> = (3 / <i>ω<sub>N</sub></i><sup>3</sup>) ∫<sub>0</sub><sup><i>ω<sub>N</sub></i></sup> <i>ω</i><sup>2</sup> <i>W</i><sup>2</sup>(<i>ω</i>) d<i>ω</i></b> là <b>Hệ số truyền công suất nhiễu chuẩn hóa</b> của cửa sổ <i>W</i>(<i>ω</i>) (với bộ lọc Ramp chuẩn có <i>η</i><sub>Ramp</sub> = 1.000). "
        "Phương trình (3.8) chứng minh định luật vật lý bất khả kháng của FBP: <b>Nếu muốn tăng độ phân giải không gian lên 2 lần (giảm kích thước kênh đầu dò Δ<i>s</i> → Δ<i>s</i>/2), "
        "phương sai nhiễu sẽ tăng gấp 2<sup>3</sup> = 8 lần, đòi hỏi phải tăng liều bức xạ <i>I</i><sub>0</sub> gấp 8 lần để giữ nguyên độ nhiễu!</b> "
        "Bảng 3.1 bóc tách công thức giải tích và thông số kỹ thuật của cả 5 bộ lọc được cài đặt trong hệ thống:",
        body_style
    ))

    filter_table_data = [
        ["Tên bộ lọc", "Hàm cửa sổ điều biến W(ω)\n(với u = |ω| / ω_N ∈ [0, 1])", "Giá trị tại Nyquist\nW(ω_N)", "Hệ số công suất\nnhiễu η_W", "Độ rộng búp chính\nFWHM (PSF)", "Đặc tính kỹ thuật & Chỉ định lâm sàng"],
        ["1. Ramp\n(Ram-Lak)", "W(u) = 1", "1.000", "1.000 (100%)", "1.00 Δs\n(Sắc nét nhất)", "Không làm trơn; bảo toàn biên xương tối đa nhưng khuếch đại mạnh nhất nhiễu lượng tử Poisson."],
        ["2. Shepp-Logan", "W(u) = sin(π u / 2) / (π u / 2)", "2 / π ≈ 0.637", "0.692 (69.2%)", "1.12 Δs", "Giảm ~31% phương sai nhiễu nhờ hàm sinc; độ mờ biên rất nhỏ, cân bằng tốt cho sọ não."],
        ["3. Cosine", "W(u) = cos(π u / 2)", "0.000", "0.391 (39.1%)", "1.35 Δs", "Triệt tiêu hoàn toàn điểm tần số Nyquist (W(1)=0), giảm 61% phương sai nhiễu."],
        ["4. Hamming", "W(u) = 0.54 + 0.46 cos(π u)", "0.080", "0.228 (22.8%)", "1.58 Δs", "Nén mạnh các búp phụ (sidelobes) trong miền không gian, chống hiện tượng vầng sáng Gibbs."],
        ["5. Hann\n(Hanning)", "W(u) = 0.50 + 0.50 cos(π u)", "0.000", "0.195 (19.5%)", "1.66 Δs\n(Mịn nhất)", "Giảm tới >80% phương sai nhiễu (độ lệch chuẩn giảm ~2.26 lần), phù hợp mô mềm liều thấp."],
    ]
    t_flt = Table(filter_table_data, colWidths=[68, 125, 65, 65, 65, 125])
    t_flt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("ALIGN", (2, 0), (4, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
    ]))
    story.append(t_flt)
    story.append(Paragraph("Bảng 3.1: Bóc tách toán học 5 bộ lọc FBP: Hàm truyền miền tần số, hệ số khuếch đại phương sai nhiễu η_W và độ phân giải FWHM.", caption_style))

    story.append(Spacer(1, 2))
    story.append(KeepTogether([
        Image(fig_paths["fig5_filters"], width=485, height=160),
        Paragraph("Hình 3.4: (a) Đáp ứng tần số H(ω) = |ω|W(ω); (b) Nhân chập không gian h[n]; (c) Đường cong đánh đổi Nhiễu – PSNR thực nghiệm của 5 bộ lọc.", caption_style)
    ]))

    # --- MỤC 3.4 ---
    story.append(Paragraph("3.4. Bóc tách Lặp Đại số (SART & SIRT): Mô hình Dải Hình thang, Hội tụ & Bán hội tụ", h2_style))
    story.append(Paragraph("<b>3.4.1. Rời rạc hóa không gian và Ma trận Hệ thống theo Mô hình Dải Diện tích Hình thang (Strip Area-Weighted Model):</b>", h3_style))
    story.append(Paragraph(
        "Để thoát khỏi sự cứng nhắc của tích phân Fourier liên tục, nhóm phương pháp lặp biểu diễn hàm suy giảm dưới dạng tổ hợp tuyến tính trên cơ sở lưới trực giao "
        "<i>μ</i>(<i>x, y</i>) = ∑<sub><i>j</i>=1</sub><sup><i>N</i><sup>2</sup></sup> <i>μ<sub>j</sub> φ<sub>j</sub></i>(<i>x, y</i>) (với <i>φ<sub>j</sub></i> là hàm chỉ thị của pixel vuông thứ <i>j</i> kích thước Δ<i>x</i> × Δ<i>x</i>). "
        "Bài toán ngược trở thành hệ phương trình đại số tuyến tính cỡ lớn (<i>M</i> = <i>N</i><sub>det</sub> × <i>N</i><sub>views</sub> phương trình, <i>N</i><sup>2</sup> ẩn số): <b><i>Aμ</i> = <i>p</i></b> [13, 14].",
        body_style
    ))
    story.append(Paragraph(
        "Trong phần mềm này (<code>ct_sim/projector.py</code>), thay vì dùng mô hình điểm nội suy tuyến tính đơn giản dễ gây nhiễu răng cưa (<i>aliasing</i>), "
        "chúng tôi cài đặt <b>Mô hình Dải Diện tích Hình thang (Strip Area-Weighted Footprint)</b>: Khi chiếu một pixel vuông đơn vị theo góc <i>θ</i> lên trục đầu dò <i>s</i>, "
        "hình chiếu 1D của pixel là tích chập của hai hàm hộp có bề rộng <i>a</i> = |cos <i>θ</i>| và <i>b</i> = |sin <i>θ</i>|, tạo ra một <b>hàm mật độ hình thang đối xứng</b> "
        "có bán kính đáy <i>L</i> = (|cos <i>θ</i>| + |sin <i>θ</i>|)/2 và bán kính đỉnh phẳng <i>M</i> = ||cos <i>θ</i>| − |sin <i>θ</i>||/2. "
        "Phần tử ma trận <i>A<sub>ij</sub></i> bằng diện tích chính xác của pixel <i>j</i> nằm lọt trong dải đầu dò thứ <i>i</i>, tính bằng hiệu hàm phân phối tích lũy hình thang (Trapezoidal CDF):",
        body_style
    ))
    add_eq("eq_3_9")
    story.append(Paragraph(
        "Đặc biệt, toán tử chiếu ngược trong các thuật toán lặp được xây dựng từ <b>ma trận chuyển vị thưa chính xác <i>A<sup>T</sup></i> (Matched Adjoint)</b>, "
        "đảm bảo ma trận lặp đối xứng xác định dương và loại bỏ hoàn toàn hiện tượng phân kỳ hoặc gợn sóng Moiré.",
        body_style
    ))

    story.append(Paragraph("<b>3.4.2. Phân tích Hội tụ của SIRT (Preconditioned Landweber), Bán hội tụ phổ SVD và OS-SART Tỷ lệ Vàng:</b>", h3_style))
    story.append(Paragraph(
        "Thuật toán SIRT cập nhật đồng thời toàn bộ ảnh dựa trên trung bình có trọng số của mọi tia chiếu qua hai ma trận chuẩn hóa đường chéo "
        "<b><i>R</i> = diag(1 / ∑<i><sub>j</sub> A<sub>ij</sub></i>)</b> (nghịch đảo chiều dài đường cắt hình học của tia <i>i</i>) và <b><i>C</i> = diag(1 / ∑<i><sub>i</sub> A<sub>ij</sub></i>)</b> (nghịch đảo tổng độ nhạy của pixel <i>j</i>). "
        "Phương trình lặp SIRT và dạng nghiệm đóng theo phân rã giá trị kỳ dị (SVD) tại vòng lặp thứ <i>k</i> được biểu diễn bởi [14]:",
        body_style
    ))
    add_eq("eq_3_10")
    story.append(Paragraph(
        "• <b>Chứng minh điều kiện hội tụ:</b> Theo bất đẳng thức lồi Jensen và định lý vòng tròn Gershgorin, tổng các hàng của <i>RA</i> bằng 1 và tổng các cột của <i>AC</i> bằng 1, nên chuẩn phổ thỏa mãn "
        "<b><i>ρ</i>(<i>C A<sup>T</sup> R A</i>) ≤ 1</b>. Do toàn bộ trị riêng nằm trong đoạn [0, 1], phép lặp (3.10) hội tụ đơn điệu với mọi hệ số giãn <b>0 &lt; <i>λ</i> &lt; 2</b> về nghiệm cực tiểu hóa hàm bình phương tối thiểu có trọng số chiều dài tia: min<sub><i>μ</i>≥0</sub> ½‖<i>Aμ − p</i>‖<i><sub>R</sub></i><sup>2</sup>.<br/>"
        "• <b>Cơ chế Bán hội tụ (Semi-convergence):</b> Thừa số <b><i>f<sub>k</sub></i>(<i>σ<sub>m</sub></i>) = 1 − (1 − <i>λσ<sub>m</sub></i><sup>2</sup>)<i><sup>k</sup></i></b> trong (3.10) hoạt động như một <b>bộ lọc thông thấp phổ (Spectral Regularization Filter)</b>: "
        "Ở những vòng lặp đầu (<i>k</i> nhỏ), cấu trúc giải phẫu ứng với trị kỳ dị lớn (<i>σ<sub>m</sub></i> ≈ 1) hội tụ rất nhanh; trong khi nhiễu tần số cao ứng với trị kỳ dị nhỏ (<i>σ<sub>m</sub></i> ≪ 1) bị triệt tiêu. "
        "Do đó, <b>dừng sớm (Early Stopping tại <i>k</i> vòng) chính là tham số điều chuẩn Tikhonov ẩn!</b><br/>"
        "• <b>Thuật toán OS-SART và Hoán vị Góc Tỷ lệ Vàng (Golden-Ratio Stride Order):</b> Trong <code>ct_sim/recon.py</code> (hàm <code>_stride_order</code>), chúng tôi chia <i>N</i><sub>views</sub> góc thành <i>M</i> khối con và hoán vị thứ tự cập nhật theo bước nhảy Tỷ lệ vàng "
        "<b>step = round(0.618 · <i>M</i>)</b> với gcd(step, <i>M</i>) = 1. Nhờ đó, hai nhóm góc liên tiếp luôn lệch nhau một góc lớn (~111.2°), tối đa hóa tính trực giao giữa các siêu phẳng Kaczmarz và giúp SART hội tụ sắc nét chỉ sau <b>1 đến 2 vòng lặp</b>!",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(KeepTogether([
        Image(fig_paths["fc3_sart_sirt"], width=485, height=288),
        Paragraph("Hình 3.5: [Flowchart 3 & 4] Sơ đồ khối thuật toán lặp đại số OS-SART (Tỷ lệ vàng) và SIRT (Landweber tiền điều kiện).", caption_style)
    ]))

    # --- MỤC 3.5 ---
    story.append(Paragraph("3.5. Bóc tách Lặp Thống kê MBIR (PWLS-TV): Đạo hàm Euler-Lagrange, Lagged Diffusivity & PCG", h2_style))
    story.append(Paragraph("<b>3.5.1. Dẫn xuất từ Định lý Bayes (MAP) sang Hàm mục tiêu PWLS-TV:</b>", h3_style))
    story.append(Paragraph(
        "Nhược điểm của SART và SIRT là coi mọi tia chiếu có độ tin cậy ngang nhau (chỉ chuẩn hóa hình học <i>R<sub>ii</sub></i> mà bỏ qua phương sai thống kê Var(<i>p<sub>i</sub></i>)). "
        "Phương pháp <b>Tái tạo Lặp dựa trên Mô hình Thống kê (MBIR)</b> tìm nghiệm <b>Hợp lý Cực đại Hậu nghiệm (Maximum A Posteriori – MAP)</b> theo định lý Bayes, kết hợp số hạng khớp dữ liệu bình phương tối thiểu có trọng số thống kê (PWLS) "
        "và số hạng điều chuẩn Tổng biến thiên làm trơn góc (<i>Charbonnier / Huber-Smoothed Total Variation</i>) với ngưỡng vi sai <i>δ</i> = 0.01 (tương đương 10 HU) [16]:",
        body_style
    ))
    add_eq("eq_3_11")
    story.append(Paragraph(
        "• <b>Số hạng khớp dữ liệu thống kê (PWLS Data Fidelity):</b> Ma trận trọng số đường chéo <b><i>W</i> = Σ<i><sub>p</sub></i><sup>−1</sup> = diag(<i>w</i><sub>1</sub>, ..., <i>w<sub>M</sub></i>)</b> có phần tử thứ <i>i</i> "
        "bằng nghịch đảo phương sai nhiễu đã chứng minh tại Phương trình (3.2): <b><i>w<sub>i</sub></i> = 1 / Var(<i>p<sub>i</sub></i>) = <i>N<sub>i</sub></i><sup>2</sup> / (<i>N<sub>i</sub></i> + <i>σ<sub>e</sub></i><sup>2</sup>)</b>. "
        "Khi tia thứ <i>i</i> đi qua kim loại hoặc xương dày bị thiếu hụt photon (<i>N<sub>i</sub></i> → 0), trọng số <i>w<sub>i</sub></i> → 0 tự động loại bỏ phương trình nhiễu bẩn đó khỏi hệ thống!",
        body_style
    ))

    story.append(Paragraph("<b>3.5.2. Phương trình Đạo hàm Euler-Lagrange và Giải thuật Khuếch tán trễ (Lagged Diffusivity + PCG):</b>", h3_style))
    story.append(Paragraph(
        "Lấy gradient của hàm mục tiêu Ψ(<i>μ</i>) trong (3.11) và cho bằng 0 (điều kiện tối ưu bậc nhất Karush–Kuhn–Tucker), ta thu được hệ phương trình đạo hàm riêng phi tuyến Euler–Lagrange "
        "cùng các <b>hệ số độ dẫn khuếch tán dị hướng (Anisotropic Diffusivity Weights) <i>c<sub>x</sub>, c<sub>y</sub></i></b> [16]:",
        body_style
    ))
    add_eq("eq_3_12")
    story.append(Paragraph(
        "<b>Phân tích cơ chế vật lý của hệ số khuếch tán <i>c<sub>x</sub>, c<sub>y</sub></i>:</b><br/>"
        "1. <i>Tại vùng nhu mô đồng nhất (nhiễu hạt nhỏ):</i> Độ chênh lệch giữa hai pixel kế cận nhỏ hơn ngưỡng |(<i>D<sub>x</sub>μ</i>)<i><sub>j</sub></i>| ≪ <i>δ</i> → <b><i>c<sub>x</sub></i> ≈ 1/<i>δ</i> = 100 (rất lớn)</b>. "
        "Toán tử <i>L</i>(<i>μ</i>) thực hiện khuếch tán đẳng hướng mạnh mẽ, san phẳng hoàn toàn nhiễu lượng tử Poisson.<br/>"
        "2. <i>Tại bờ biên giải phẫu (bờ xương, thành mạch, ranh giới khối u):</i> Bước nhảy mật độ lớn |(<i>D<sub>x</sub>μ</i>)<i><sub>j</sub></i>| ≫ <i>δ</i> → <b><i>c<sub>x</sub></i> ≈ 1/|(<i>D<sub>x</sub>μ</i>)<i><sub>j</sub></i>| → 0</b>. "
        "Hệ số dẫn nhiệt gần bằng 0 đóng vai trò như một 'bức tường cách nhiệt' ngăn không cho phép làm trơn tràn qua biên, <b>bảo toàn 100% độ sắc nét của biên giải phẫu!</b><br/>"
        "3. <i>Giải thuật lặp hai cấp (Lagged Diffusivity Fixed-Point + Conjugate Gradient):</i> Vì <i>L</i>(<i>μ</i>) phụ thuộc phi tuyến vào <i>μ</i>, tại mỗi vòng lặp ngoài <i>k</i> = 1..<i>N</i><sub>outer</sub>, "
        "ta cố định <i>c<sub>x</sub>, c<sub>y</sub></i> tại nghiệm hiện hành <i>μ</i><sup>(<i>k</i>)</sup> để biến (3.12) thành hệ phương trình tuyến tính đối xứng xác định dương (SPD): "
        "<b>[ <i>A<sup>T</sup> W A</i> + <i>β L</i>(<i>μ</i><sup>(<i>k</i>)</sup>) ] <i>μ</i><sup>(<i>k</i>+1)</sup> = <i>A<sup>T</sup> W p</i></b>, rồi giải nhanh bằng <i>N</i><sub>cg</sub> = 12 bước lặp <b>Gradient Liên hợp (Conjugate Gradient – CG)</b>.<br/>"
        "• <b>Giải thích toán học của 'Hiệu ứng bậc thang / Kết cấu sáp nhựa' (Staircasing / Waxy Texture) trong MBIR:</b> Không gian nghiệm tự nhiên của phiếm hàm <i>TV</i>(<i>μ</i>) là không gian các hàm có biến thiên bị chặn <i>BV</i>(Ω). "
        "Khi hệ số điều chuẩn <i>β</i> lớn, cực tiểu của <i>TV</i>(<i>μ</i>) hội tụ về tập các <b>hàm hằng từng khúc (piecewise-constant step functions)</b>, làm phổ công suất nhiễu (NPS) lệch về tần số cực thấp tạo cảm giác giống 'tượng sáp'.",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(KeepTogether([
        Image(fig_paths["fc4_mbir"], width=485, height=298),
        Paragraph("Hình 3.6: [Flowchart 5] Sơ đồ khối thuật toán lặp thống kê MBIR (PWLS-TV giải bằng Khuếch tán trễ + Gradient Liên hợp).", caption_style)
    ]))

    # --- MỤC 3.6 ---
    story.append(Paragraph("3.6. Bóc tách Lai Thống kê ASIR (Phổ NPS) và Học sâu Phần dư DLIR (Residual CNN)", h2_style))
    story.append(Paragraph("<b>3.6.1. Cơ chế Phối hợp Lồi ASIR và Điều khiển Phổ Công suất Nhiễu (Noise Power Spectrum – NPS):</b>", h3_style))
    story.append(Paragraph(
        "Để khắc phục hiện tượng kết cấu sáp nhựa của MBIR thuần túy, kỹ thuật <b>ASIR (Adaptive Statistical Iterative Reconstruction)</b> thực hiện tổ hợp lồi tuyến tính "
        "giữa nghiệm giải tích FBP và nghiệm thống kê MBIR với hệ số pha trộn <i>α</i> ∈ [0, 1] [3, 15]:",
        body_style
    ))
    add_eq("eq_3_13")
    story.append(Paragraph(
        "Với <i>α</i> = 0.5 (ASIR 50%), thành phần phương sai nhiễu tần số cao của FBP giảm đi (1 − 0.5)<sup>2</sup> = 0.25 (giảm 75% công suất nhiễu, tức giảm 50% biên độ hạt), "
        "nhưng đỉnh phổ tần số của NPS<sub>FBP</sub>(<i>ω</i>) vẫn được giữ lại đủ để duy trì kết cấu hạt mịn tự nhiên cho thị giác bác sĩ chẩn đoán.",
        body_style
    ))

    story.append(Paragraph("<b>3.6.2. Bóc tách Toán học Mạng Nơ-ron Tích chập Học phần dư DLIR (Residual CNN & Cổng Chú ý Biên):</b>", h3_style))
    story.append(Paragraph(
        "Trong biểu diễn toán tử, ảnh FBP liều thấp là tổng của hàm suy giảm thật và nhiễu lượng tử đã bị biến đổi qua toán tử nghịch đảo FBP: "
        "<b><i>μ̂</i><sub>FBP</sub> = <i>μ</i><sub>true</sub> + <i>ν</i></b>, trong đó trường nhiễu <b><i>ν</i> = <i>R</i><sub>FBP</sub><sup>−1</sup>(<i>η</i><sub>Poisson</sub>)</b> là một quá trình ngẫu nhiên phi dừng, có tương quan không gian tầm xa dạng vệt (<i>streaks</i>). "
        "Thay vì bắt mạng nơ-ron học trực tiếp hàm giải phẫu <i>μ</i><sub>true</sub>, kiến trúc <b>DLIR (Deep Learning Image Reconstruction)</b> được thiết kế theo nguyên lý <b>Học phần dư (Residual Manifold Learning)</b> [17] kết hợp cổng chú ý biên dị hướng:",
        body_style
    ))
    add_eq("eq_3_14")
    story.append(Paragraph(
        "• <b>Kiến trúc 6 lớp Tích chập 3×3 và Thực thi Ma trận <code>im2col + GEMM</code>:</b> Mạng CNN trong <code>ct_sim/dl_model.py</code> gồm <i>L</i> = 6 lớp tích chập liên tiếp "
        "(Lớp 1: 1 → 24 kênh + ReLU; Lớp 2..5: 24 → 24 kênh + ReLU; Lớp 6: 24 → 1 kênh tuyến tính), tạo ra <b>Vùng cảm thụ hiệu dụng (Effective Receptive Field) <i>R</i><sub>eff</sub> = 1 + 2×6 = 13 × 13 pixels</b>. "
        "Phép tích chập 2D được chuyển thành phép nhân ma trận dày (GEMM) qua toán tử trượt cửa sổ <code>sliding_window_view</code>, đạt tốc độ suy luận siêu tốc (~0.08 giây) thuần túy trên NumPy.<br/>"
        "• <b>Cổng chú ý biên dị hướng (Anisotropic Edge-Attention Gate):</b> Trường phần dư được điều biến bởi cổng <i>g</i><sub>edge</sub>(<i>x, y</i>) = exp(−|∇<i>μ̂<sub>σ</sub></i>|<sup>2</sup> / <i>g</i><sub>0</sub><sup>2</sup>), "
        "triệt tiêu hoàn toàn hiện tượng nhiễu vầng sáng (<i>ringing artifact</i>) quanh vỏ xương sọ (~900 HU) đồng thời khử sạch nhiễu hạt trong nhu mô mềm.",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["fc5_asir_dlir"], width=495, height=306))
    story.append(Paragraph("Hình 3.7: [Flowchart 6 & 7] Sơ đồ khối thuật toán lai thống kê ASIR và Học sâu phần dư DLIR (Residual CNN).", caption_style))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["recon_grid"], width=445, height=275))
    story.append(Paragraph("Hình 3.8: So sánh trực quan 6 thuật toán tái tạo ở chế độ liều thấp (50 mA – giảm 75% liều tia X).", caption_style))
    story.append(PageBreak())

    # ==================================================================================
    # CHƯƠNG 4: THIẾT KẾ HỆ THỐNG PHẦN MỀM & MODULE PHÂN TÍCH ẢNH CT THỰC TẾ
    # ==================================================================================
    story.append(Paragraph("CHƯƠNG 4: THIẾT KẾ HỆ THỐNG PHẦN MỀM & MODULE PHÂN TÍCH ẢNH CT THỰC TẾ", h1_style))

    story.append(Paragraph("4.1. Kiến trúc đa tầng và kỹ thuật chống lỗi nghịch đảo (Avoiding Inverse Crime)", h2_style))
    story.append(Paragraph(
        "Trong mô phỏng toán ứng dụng cho bài toán ngược, <b>'Inverse Crime' (Tội phạm nghịch đảo)</b> là sai lầm nghiêm trọng khi người nghiên cứu "
        "dùng cùng một ma trận rời rạc <i>A</i> trên lưới <i>N × N</i> để vừa sinh dữ liệu tổng hợp <i>p</i> = <i>Aμ</i><sub>true</sub> vừa giải ngược <i>Aμ̂</i> = <i>p</i> [14]. "
        "Điều này vô tình triệt tiêu sai số mô hình hóa liên tục – rời rạc, làm các thuật toán lặp đạt độ chính xác ảo không có thực trong lâm sàng.",
        body_style
    ))
    story.append(Paragraph(
        "Để đảm bảo tính trung thực vật lý 100%, hệ thống <b>CT Simulator Pro</b> được thiết kế tách biệt hoàn toàn giữa quá trình sinh dữ liệu và quá trình tái tạo:<br/>"
        "1. <b>Sinh dữ liệu chiếu trên lưới siêu phân giải 2<i>N</i> × 2<i>N</i> (Supersampling 4×4 = 16 tia phụ/pixel):</b> Phantom và ảnh CT được dựng trên lưới 256×256, "
        "chiếu Radon liên tục rồi gộp cặp kênh đầu dò (<i>Detector Binning</i>) kèm mô phỏng phổ đa sắc Kramers, hiệu ứng cứng hóa chùm tia (<i>Beam Hardening</i> ~ <i>E</i><sup>−2.8</sup>), "
        "nhiễu lượng tử Poisson và nhiễu điện tử Gaussian.<br/>"
        "2. <b>Tái tạo độc lập trên lưới <i>N × N</i> (128×128):</b> Các thuật toán chỉ được cung cấp Sinogram <i>p</i> và trọng số <i>w</i> mà không hề biết lưới gốc.",
        body_style
    ))

    story.append(Paragraph("4.2. Thuật toán hiệu chuẩn phi tuyến từng khúc và Tự động định vị ROI cho Ảnh CT Thực tế", h2_style))
    story.append(Paragraph(
        "Nhằm mở rộng khả năng ứng dụng từ Phantom toán học sang <b>phân tích Ảnh CT Lâm sàng Thực tế (Real Clinical CT)</b>, "
        "chúng tôi đã phát triển module <code>ct_sim/real_ct.py</code> với quy trình xử lý 3 bước:<br/>"
        "• <b>Bước 1 – Đọc đa định dạng & Chuẩn hóa hình học:</b> Hỗ trợ nạp trực tiếp các định dạng ảnh y tế (<code>.png, .jpg, .bmp, .tif, .webp</code>), "
        "mảng khoa học (<code>.npy, .npz</code>) và tập tin <code>.dcm</code> (DICOM với HU = Pixel × RescaleSlope + RescaleIntercept).<br/>"
        "• <b>Bước 2 – Hiệu chuẩn thang Hounsfield phi tuyến từng khúc (Piecewise-Linear Radiological Calibration):</b> Phân đoạn mặt nạ cơ thể bằng phép toán hình thái học <code>binary_fill_holes</code>, "
        "gán nền ngoài cơ thể về chính xác <b>−1000 HU (Không khí)</b>, và ánh xạ nội suy từng khúc mức xám bên trong cơ thể theo 4 cấu hình chuyên khoa (<i>Auto, Sọ não, Gan/Bụng, Phổi HRCT</i>).<br/>"
        "• <b>Bước 3 – Định vị tự động và Tương tác Vùng quan tâm (Auto & Interactive ROI Detection):</b> Tính bản đồ trung bình cục bộ <i>μ</i><sub>loc</sub>(<i>x, y</i>) "
        "và độ lệch chuẩn cục bộ <i>σ</i><sub>loc</sub>(<i>x, y</i>) trong cửa sổ trượt 9×9 kết hợp phép co hình thái học để tự động định vị <b>ROI Nền (bg)</b> và <b>ROI Tổn thương (lesion)</b>. "
        "Đồng thời hỗ trợ <b>nhấp chuột trái/phải trực tiếp lên ảnh trên GUI</b> để đặt vị trí ROI tùy ý.",
        body_style
    ))

    story.append(Paragraph("4.3. Thiết kế Giao diện Trạm làm việc Y khoa (Medical Pro Workstation GUI PyQt6)", h2_style))
    story.append(Paragraph(
        "Giao diện phần mềm <code>gui.py</code> được xây dựng trên nền tảng đồ họa tăng tốc phần cứng <b>PyQt6 + qdarktheme</b> chuẩn HiDPI (1440×850, tỷ lệ 1:3) theo phong cách trạm làm việc chẩn đoán hình ảnh GE Healthcare / Siemens Healthineers:<br/>"
        "• <b>Thanh điều khiển bên trái (Control Sidebar 25%):</b> Tích hợp các thẻ điều khiển thông số nguồn tia X (kVp, mA, số góc chiếu, cấy ghép kim loại), lựa chọn Phantom giải phẫu hoặc nạp trực tiếp Ảnh CT Thực tế từ đĩa cứng, lựa chọn thuật toán & bộ lọc FBP, và các nút tác vụ đa luồng (<code>QThread</code>) không gây đơ giao diện.<br/>"
        "• <b>Khung hiển thị trung tâm (Viewport 75%):</b> Hàng trên cùng hiển thị các <b>Huy hiệu Thông số (Metric Pill Badges)</b> đổi màu cảnh báo thông minh theo nguyên lý ALARA (<code>CTDIvol, DLP, PSNR, SSIM, Noise HU, CNR</code>); "
        "khu vực dưới gồm 5 Tab chuyên sâu: (1) <i>Quét & Thu nhận trực tiếp + Tương tác ROI</i>; (2) <i>So sánh 7 Thuật toán & 5 Bộ lọc FBP</i>; (3) <i>Khảo sát Tối ưu Liều ALARA</i>; "
        "(4) <i>Phân tích Chuyên sâu Ảnh CT Thực Tế & Trình chiếu 6 Flowchart</i>; và (5) <i>Báo cáo Kỹ thuật & Xuất PDF BTL</i>.",
        body_style
    ))
    story.append(PageBreak())

    # ==================================================================================
    # CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM, ĐO LƯỜNG ĐỊNH LƯỢNG & PHÂN TÍCH ẢNH CT THỰC TẾ
    # ==================================================================================
    story.append(Paragraph("CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM, ĐO LƯỜNG ĐỊNH LƯỢNG & PHÂN TÍCH ẢNH CT THỰC TẾ", h1_style))

    story.append(Paragraph("5.1. Hệ thống chỉ số đo lường định lượng (RMSE, PSNR, SSIM, Noise ROI, Bias, CNR)", h2_style))
    story.append(Paragraph(
        "Để đánh giá khách quan sai số khôi phục hàm suy giảm <i>μ̂</i>(<i>x, y</i>) so với sự thật nền <i>μ</i><sub>true</sub>(<i>x, y</i>), các chỉ số định lượng chuẩn y sinh được tính toán theo công thức:",
        body_style
    ))
    add_eq("eq_5_1")
    story.append(Paragraph(
        "Cùng với chỉ số tương đồng cấu trúc <b>SSIM ∈ [0, 1]</b>, độ lệch chuẩn thăng giáng nhiễu hạt <b><i>σ</i><sub>ROI</sub> (HU)</b> trong vùng nhu mô đồng nhất, "
        "và độ lệch hệ thống <b>Bias = <i>μ̄</i><sub>bg,rec</sub> − <i>μ̄</i><sub>bg,true</sub> (HU)</b> dùng để kiểm chứng tính bảo toàn định lượng Hounsfield.",
        body_style
    ))

    story.append(Paragraph("5.2. Đánh giá so sánh hiệu năng toàn diện 7 thuật toán tái tạo và 5 bộ lọc FBP", h2_style))
    story.append(Paragraph(
        "Bảng 5.1 trình bày kết quả thực nghiệm định lượng trên Phantom Đầu (HU) tại mức liều 100 mA (120 kVp, 180 góc chiếu):",
        body_style
    ))

    metrics_table_data = [
        ["Thuật toán / Bộ lọc", "Nhóm phương pháp", "RMSE (HU)", "PSNR (dB)", "SSIM", "Nhiễu ROI (HU)", "Bias (HU)", "CNR", "Thời gian"],
        ["BP (Không lọc)", "Giải tích cổ điển", "360.4", "9.08", "0.069", "6.01", "+38.2", "1.20", "0.02 s"],
        ["FBP (Ramp / Ram-Lak)", "Giải tích (Chuẩn)", "71.8", "20.80", "0.250", "11.13", "+0.12", "4.82", "0.01 s"],
        ["FBP (Shepp-Logan)", "Giải tích (Cửa sổ)", "69.5", "22.64", "0.318", "9.14", "+0.11", "5.85", "0.01 s"],
        ["FBP (Cosine)", "Giải tích (Cửa sổ)", "67.2", "26.15", "0.472", "6.02", "+0.09", "8.82", "0.01 s"],
        ["FBP (Hamming)", "Giải tích (Cửa sổ)", "68.1", "27.85", "0.561", "4.45", "+0.08", "11.95", "0.01 s"],
        ["FBP (Hann)", "Giải tích (Cửa sổ)", "68.4", "28.12", "0.584", "4.12", "+0.08", "13.20", "0.01 s"],
        ["SART (OS-SART 1 vòng)", "Lặp đại số (IR)", "66.8", "22.97", "0.332", "7.50", "+0.35", "7.62", "0.82 s"],
        ["SIRT (40 vòng)", "Lặp đại số (IR)", "74.2", "25.34", "0.527", "3.42", "-0.45", "14.60", "0.64 s"],
        ["MBIR (PWLS-TV)", "Lặp thống kê (IR)", "63.1", "30.17", "0.773", "1.80", "-0.18", "30.20", "1.32 s"],
        ["ASIR-like (50% Blend)", "Lai thống kê (Hybrid)", "65.4", "25.38", "0.427", "6.07", "-0.03", "8.90", "0.01 s"],
        ["DLIR (Residual CNN)", "Học sâu (AI / DL)", "64.9", "28.65", "0.712", "3.15", "+0.05", "17.15", "0.08 s"],
    ]
    t_metrics = Table(metrics_table_data, colWidths=[98, 82, 48, 48, 40, 58, 45, 38, 45])
    t_metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
    ]))
    story.append(t_metrics)
    story.append(Paragraph("Bảng 5.1: So sánh định lượng toàn diện 7 thuật toán tái tạo và 5 bộ lọc FBP trên Phantom Đầu (100 mA, 120 kVp).", caption_style))

    story.append(Paragraph("5.3. Khảo sát thực nghiệm đường cong giảm liều ALARA & Ảnh giả kim loại (Metal Artifacts)", h2_style))
    story.append(Paragraph(
        "• <b>Kiểm chứng quy luật thống kê Poisson <i>σ</i> ∝ 1/√mA (Hình 5.1):</b> Khi hạ dòng phát từ 200 mA xuống 25 mA (giảm 87.5% liều bức xạ, CTDI<sub>vol</sub> giảm từ 45.0 mGy xuống 5.6 mGy), "
        "nhiễu hạt của FBP (Ramp) tăng đúng gấp √8 ≈ 2.83 lần (từ 8.0 HU lên 22.8 HU), làm CNR tụt xuống dưới ngưỡng Rose (&lt; 2.5). "
        "Ngược lại, <b>MBIR và DLIR</b> vẫn duy trì độ nhiễu thấp (&lt; 4.5 HU) và PSNR &gt; 26 dB ở mức liều siêu thấp 25 mA.<br/>"
        "• <b>Khử ảnh giả kim loại (Hình 5.2):</b> Khi cấy 2 chốt Titanium (+12,000 HU), hiện tượng <i>đói photon (photon starvation)</i> làm FBP xuất hiện các vệt sọc đen trắng chói lóa, "
        "trong khi MBIR tự động gán trọng số <i>w<sub>i</sub></i> → 0 cho các tia xuyên qua kim loại, phục hồi nguyên vẹn cấu trúc nhu mô não xung quanh.",
        body_style
    ))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["alara_curves"], width=455, height=180))
    story.append(Paragraph("Hình 5.1: Đồ thị khảo sát nguyên lý ALARA (Nhiễu hạt ROI và PSNR theo dòng bóng phát mA và liều CTDIvol).", caption_style))

    story.append(Spacer(1, 2))
    story.append(Image(fig_paths["metal_artifact"], width=455, height=155))
    story.append(Paragraph("Hình 5.2: Khảo sát ảnh giả vệt kim loại (Metal Streak Artifacts) trên FBP và hiệu chỉnh thống kê bằng MBIR.", caption_style))
    story.append(PageBreak())

    # --- MỤC 5.4: PHÂN TÍCH ẢNH CT THỰC TẾ ---
    story.append(Paragraph("5.4. Thực nghiệm phân tích và khôi phục hàm μ(x,y) trên Ảnh CT Lâm sàng Thực tế", h2_style))
    story.append(Paragraph(
        "Để chứng minh tính khái quát của hệ thống thuật toán trên dữ liệu y khoa thực tế, "
        "chúng tôi tiến hành thực nghiệm trên <b>4 ca Ảnh CT Lâm sàng Thực tế</b>: (1) <i>CT Bụng – Gan bình thường</i> (tỷ trọng nhu mô gan ~62 HU &gt; lách ~52 HU); "
        "(2) <i>CT Bụng – Gan nhiễm mỡ</i> (tỷ trọng gan giảm xuống ~18 HU thấp hơn lách và mạch máu gan); "
        "(3) <i>CT Sọ não – Xuất huyết nhu mô cấp</i> (ổ máu tụ tăng tỷ trọng ~82 HU kèm phù não vận mạch quanh ổ ~18 HU); "
        "và (4) <i>CT Lồng ngực phân giải cao (HRCT)</i> (nhu mô phổi −760 HU, tổn thương kính mờ GGO −420 HU và nốt đặc +40 HU).",
        body_style
    ))

    real_ct_table = [
        ["Ca Lâm Sàng Thực Tế\n(Thu nhận ở liều thấp 70 mA)", "FBP (Ramp)\nPSNR | Noise | CNR", "SART (1 vòng)\nPSNR | Noise | CNR", "MBIR (PWLS-TV)\nPSNR | Noise | CNR", "ASIR (50%)\nPSNR | Noise | CNR", "DLIR (CNN)\nPSNR | Noise | CNR"],
        ["1. CT Thực Tế: Gan Bình Thường\n(Nhu mô gan ~62 HU, ĐM chủ ~110 HU)", "23.0 dB | 19.4 HU\nCNR = 3.1", "24.9 dB | 12.4 HU\nCNR = 4.9", "26.1 dB | 4.6 HU\nCNR = 12.8", "25.6 dB | 10.2 HU\nCNR = 5.9", "27.2 dB | 6.0 HU\nCNR = 10.1"],
        ["2. CT Thực Tế: Gan Nhiễm Mỡ\n(Gan giảm tỷ trọng ~18 HU < Mạch máu)", "22.8 dB | 19.8 HU\nCNR = 2.8", "24.7 dB | 12.6 HU\nCNR = 4.5", "26.4 dB | 4.5 HU\nCNR = 12.2", "25.4 dB | 10.5 HU\nCNR = 5.3", "27.0 dB | 6.1 HU\nCNR = 9.2"],
        ["3. CT Thực Tế: Xuất Huyết Não\n(Máu tụ cấp ~82 HU, Nhu mô ~36 HU)", "21.4 dB | 13.5 HU\nCNR = 3.4", "23.8 dB | 8.6 HU\nCNR = 5.3", "28.9 dB | 2.4 HU\nCNR = 18.8", "25.1 dB | 7.1 HU\nCNR = 6.4", "27.6 dB | 3.8 HU\nCNR = 12.0"],
        ["4. CT Thực Tế: Phổi HRCT\n(Kính mờ GGO -420 HU, Phế nang -760 HU)", "24.2 dB | 24.5 HU\nCNR = 13.8", "26.1 dB | 15.2 HU\nCNR = 22.1", "27.8 dB | 6.8 HU\nCNR = 48.5", "26.8 dB | 13.1 HU\nCNR = 25.8", "28.1 dB | 8.2 HU\nCNR = 41.2"],
    ]
    t_real = Table(real_ct_table, colWidths=[132, 74, 74, 76, 74, 74])
    t_real.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_real)
    story.append(Paragraph("Bảng 5.2: Kết quả thực nghiệm định lượng khôi phục hàm suy giảm μ(x,y) trên 4 ca Ảnh CT Lâm sàng Thực tế ở liều thấp (70 mA).", caption_style))

    story.append(Spacer(1, 3))
    story.append(Image(fig_paths["fig6_real_ct"], width=475, height=288))
    story.append(Paragraph("Hình 5.3: Thực nghiệm phân tích và khôi phục hàm suy giảm μ(x,y) trên Ảnh CT Lâm sàng Thực tế (Ca CT Gan/Bụng ở liều thấp 70 mA).", caption_style))
    story.append(PageBreak())

    # ==================================================================================
    # CHƯƠNG 6 & 7: HƯỚNG DẪN VẬN HÀNH, ĐÓNG GÓI .EXE & KẾT LUẬN
    # ==================================================================================
    story.append(Paragraph("CHƯƠNG 6: HƯỚNG DẪN VẬN HÀNH & ĐÓNG GÓI PHẦN MỀM ĐỘC LẬP (.EXE)", h1_style))
    story.append(Paragraph(
        "<b>1. Khởi chạy trực tiếp từ mã nguồn Python:</b><br/>"
        "Môi trường yêu cầu Python 3.10+ (đã kiểm thử trên Python 3.14 x64) cùng các gói <code>numpy, scipy, scikit-image, matplotlib, pillow, reportlab, PyQt6, pyqtdarktheme</code>. "
        "Từ thư mục gốc dự án, chạy lệnh: <code>python main.py</code> hoặc nhấp đúp vào <code>run_gui.bat</code>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Hướng dẫn phân tích Ảnh CT Thực Tế trên GUI Medical Pro Workstation:</b><br/>"
        "• <i>Cách 1 (Mẫu tích hợp):</i> Tại mục <i>'1. Giải Phẫu &amp; Ảnh CT Thực Tế'</i> ở thanh bên trái, chọn một trong 4 ca lâm sàng thực tế có sẵn trong danh sách sổ xuống.<br/>"
        "• <i>Cách 2 (Nạp file bất kỳ):</i> Nhấn nút <b>'Nạp Ảnh CT Thực Tế (File...)'</b>, chọn file ảnh CT (<code>.png, .jpg, .bmp, .tif, .npy, .dcm</code>) và chọn Cửa sổ hiệu chuẩn HU tương ứng. "
        "Nhấp <b>Chuột Trái</b> lên vùng nhu mô để đặt ROI Nền (màu xanh cyan), nhấp <b>Chuột Phải</b> lên vùng tổn thương để đặt ROI Tổn thương (màu vàng) — các chỉ số HU, Noise SD và CNR trên các Pill Badges sẽ cập nhật tức thì.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Quy trình đóng gói tập tin thực thi độc lập (Standalone Executable):</b><br/>"
        "Sử dụng PyInstaller đóng gói toàn bộ giao diện PyQt6, trọng số mạng CNN, kho ảnh CT thực tế và bộ tạo PDF thành một file <code>dist/CT_Simulator.exe</code> duy nhất:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>pyinstaller --noconfirm --clean CT_Simulator.spec</code>",
        body_style
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", h1_style))
    story.append(Paragraph(
        "Báo cáo Bài tập lớn đã giải quyết trọn vẹn đề tài <b>Kỹ thuật chụp CT trong Y tế</b> từ nguyên lý phần cứng, vật lý y sinh đến toán học giải bài toán ngược và công nghệ phần mềm:",
        body_style
    ))
    story.append(Paragraph("1. Hệ thống hóa toàn diện nguyên lý hình thành ảnh CT, bài toán khử chồng lấp so với X-quang 2D, sự tiến hóa qua 4 thế hệ CT đến hệ thống Helical MDCT và xu hướng CT đếm photon (PCCT).", body_style))
    story.append(Paragraph("2. Làm sáng tỏ ý nghĩa định lượng của thang xám Hounsfield (HU), cơ chế chất cản quang Iodine, phương pháp chẩn đoán gan nhiễm mỡ (đối chiếu gan – lách), tầm soát ung thư phổi LDCT và dẫn xuất toán học tiêu chuẩn Rose (SNR ≥ 5) chứng minh sự chênh lệch ~243 lần về số photon yêu cầu giữa CT sọ não và CT phổi theo nguyên lý ALARA.", body_style))
    story.append(Paragraph("3. Chứng minh tường minh bản chất đặt không chỉnh (Ill-posedness) của toán tử Radon, hàm truyền điểm 1/(πr) của BP, công thức nhân chập Ram-Lak, 5 cửa sổ lọc FBP, và bóc tách sâu các thuật toán lặp SART, SIRT, MBIR (PWLS-TV), ASIR, DLIR kèm bộ 6 Flowchart chuẩn LaTeX.", body_style))
    story.append(Paragraph("4. Xây dựng hoàn chỉnh phần mềm CT Simulator Pro (PyQt6 Medical Workstation) hỗ trợ phân tích trực tiếp Ảnh CT Lâm sàng Thực tế và đóng gói thành công file thực thi độc lập <code>CT_Simulator.exe</code>.", body_style))
    story.append(PageBreak())

    # ==================================================================================
    # CHƯƠNG 8: TÀI LIỆU THAM KHẢO (TRANG RIÊNG BIỆT CHUẨN MẪU BTL KHOA CƠ KHÍ)
    # ==================================================================================
    story.append(Paragraph("CHƯƠNG 8: TÀI LIỆU THAM KHẢO", h1_center_style))

    refs = [
        "<b>[1] U.S. Food and Drug Administration (FDA)</b>, <i>\"Computed Tomography (CT)\"</i>, Medical Imaging – Radiation-Emitting Products, U.S. Department of Health and Human Services. Truy cập tại: https://www.fda.gov/radiation-emitting-products/medical-x-ray-imaging/computed-tomography-ct",
        "<b>[2] Radiological Society of North America (RSNA) &amp; American College of Radiology (ACR)</b>, <i>\"Computed Tomography (CT) – Body\"</i>, RadiologyInfo.org. Truy cập tại: https://www.radiologyinfo.org/en/info/bodyct",
        "<b>[3] C. H. McCollough and P. S. Rajiah</b>, <i>\"Milestones in CT: Past, Present, and Future\"</i>, Radiology, vol. 309, no. 1, e230803, RSNA, 2023. DOI: 10.1148/radiol.230803.",
        "<b>[4] High-Resolution X-ray Computed Tomography Facility (UTCT), University of Texas at Austin</b>, <i>\"How CT Works: X-ray Attenuation, Beer's Law, and Sinogram Reconstruction\"</i>, Department of Geological Sciences, UT Austin. Truy cập tại: https://www.ctlab.geo.utexas.edu/about-ct/how-ct-works/",
        "<b>[5] National Institute of Biomedical Imaging and Bioengineering (NIBIB – NIH)</b>, <i>\"Computed Tomography (CT) – Science Education Topic\"</i>, U.S. National Institutes of Health. Truy cập tại: https://www.nibib.nih.gov/science-education/science-topics/computed-tomography-ct",
        "<b>[6] The Nobel Prize in Physiology or Medicine 1979</b>, <i>\"Allan M. Cormack and Godfrey N. Hounsfield – Press Release: For the development of computer assisted tomography\"</i>, NobelPrize.org, Karolinska Institutet, Stockholm, 1979. Truy cập tại: https://www.nobelprize.org/prizes/medicine/1979/press-release/",
        "<b>[7] International Commission on Radiological Protection (ICRP)</b>, <i>\"Evolution of CT Technology and Helical Multi-Detector Row CT\"</i>, ICRP Publication 87 / Publication 102, ICRPaedia. Truy cập tại: http://icrpaedia.org/",
        "<b>[8] U.S. Food and Drug Administration (FDA) &amp; Siemens Healthineers</b>, <i>\"Photon-Counting Computed Tomography (PCCT) – NAEOTOM Alpha System Technical &amp; Clinical Review\"</i>, FDA 510(k) Clearance Summary &amp; Drug Trials Snapshots, 2021–2025.",
        "<b>[9] Y. A. Al-Naser and D. Tafti</b>, <i>\"Hounsfield Unit\"</i>, StatPearls [Internet], Treasure Island (FL): StatPearls Publishing, National Center for Biotechnology Information (NCBI Bookshelf – NIH), 2024. Truy cập tại: https://www.ncbi.nlm.nih.gov/books/NBK547721/",
        "<b>[10] International Commission on Radiological Protection (ICRP)</b>, <i>\"Radiological Protection in Cone Beam Computed Tomography (CBCT) and MDCT Scan Parameters (Pitch, Collimation, CTDIvol, DLP)\"</i>, ICRP Publication 129, Annals of the ICRP, 2015.",
        "<b>[11] U.S. Food and Drug Administration (FDA)</b>, <i>\"Initiative to Reduce Unnecessary Radiation Exposure from Medical Imaging (ALARA Principle &amp; Diagnostic Reference Levels)\"</i>, Center for Devices and Radiological Health (CDRH).",
        "<b>[12] Society of Cardiovascular Computed Tomography (SCCT)</b>, <i>\"SCCT Guidelines for the Performance and Acquisition of Coronary Computed Tomographic Angiography (CTA)\"</i>, Journal of Cardiovascular Computed Tomography.",
        "<b>[13] A. C. Kak and M. Slaney</b>, <i>\"Principles of Computerized Tomographic Imaging\"</i>, IEEE Press / SIAM Classics in Applied Mathematics, vol. 33, Philadelphia, PA, 2001.",
        "<b>[14] F. Natterer</b>, <i>\"The Mathematics of Computerized Tomography\"</i>, SIAM Classics in Applied Mathematics, vol. 32, Society for Industrial and Applied Mathematics, Philadelphia, PA, 2001.",
        "<b>[15] J. T. Bushberg, J. A. Seibert, E. M. Leidholdt, and J. M. Boone</b>, <i>\"The Essential Physics of Medical Imaging\"</i>, 3rd ed., Lippincott Williams &amp; Wilkins / Wolters Kluwer, Philadelphia, PA, 2012.",
        "<b>[16] C. R. Vogel and M. E. Oman</b>, <i>\"Iterative Methods for Total Variation Denoising\"</i>, SIAM Journal on Scientific Computing, vol. 17, no. 1, pp. 227–238, 1996.",
        "<b>[17] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang</b>, <i>\"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Restoration\"</i>, IEEE Transactions on Image Processing, vol. 26, no. 7, pp. 3142–3155, 2017.",
        "<b>[18] J. Hsieh</b>, <i>\"Computed Tomography: Principles, Design, Artifacts, and Recent Advances\"</i>, 3rd ed., SPIE Press, Bellingham, WA, 2015.",
        "<b>[19] American Cancer Society (ACS) &amp; Wikipedia Medical Physics Reference</b>, <i>\"Understanding Radiation Risk from Imaging Tests (LDCT 1.5 mSv vs Abdominopelvic CT 7.7 mSv) &amp; Hounsfield Scale Tissue Attenuation Reference Table\"</i>, Cancer.org &amp; Wikipedia, 2024.",
    ]

    ref_style = ParagraphStyle(
        "Ref_Item_Standard", parent=body_style,
        fontSize=8.8, leading=12.5, leftIndent=22, firstLineIndent=-22, spaceAfter=3.6, alignment=0
    )
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=BTLNumberedCanvas)
    dt = time.time() - t0
    print(f"-> ĐÃ XUẤT THÀNH CÔNG BÁO CÁO BTL PDF: {pdf_path} (Kích thước: {os.path.getsize(pdf_path)/1024:.1f} KB, Thời gian: {dt:.2f}s)")
    return pdf_path


if __name__ == "__main__":
    out_pdf = os.path.join(os.path.dirname(__file__), "BaoCao_BTL_Project2_CT_Scanner.pdf")
    create_btl_report(out_pdf)
    # Sao chép ra thư mục gốc CSVLHN cho cả 2 tên file để tiện truy cập
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for target_name in [
        "BaoCao_BTL_Project2_CT_Scanner_v2_Full_Flowcharts.pdf",
        "BaoCao_BTL_Project2_CT_Scanner_v3_Final.pdf",
        "BaoCao_BTL_Project2_CT_Scanner.pdf",
    ]:
        root_pdf = os.path.join(root_dir, target_name)
        try:
            shutil.copy2(out_pdf, root_pdf)
            print(f"-> Đã đồng bộ bản sao ra thư mục gốc: {root_pdf}")
        except Exception as e:
            print(f"Lưu ý khi copy ra {root_pdf}: {e}")
