"""
phantoms.py - Bộ phantom giải phẫu (đơn vị Hounsfield) cho mô phỏng CT.

Mỗi phantom được định nghĩa bằng danh sách ellipse trong toạ độ chuẩn hoá [-1, 1]^2.
Ảnh được dựng ở độ phân giải cao (supersampling) rồi lấy trung bình để có biên khử răng cưa
(anti-aliasing) -> tránh "inverse crime" khi dùng cùng lưới với bộ tái tạo.

Thang Hounsfield:   HU = 1000 * (mu - mu_water) / mu_water     (air = -1000, water = 0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# (A, a, b, x0, y0, phi_deg) - A: giá trị (cộng dồn hoặc thay thế tuỳ phantom)
Ellipse = Tuple[float, float, float, float, float, float]

HU_AIR = -1000.0
HU_METAL = 12000.0  # kim loại (thép / Ti) -> vượt xa thang hiển thị, gây "photon starvation"


@dataclass
class PhantomInfo:
    """Siêu dữ liệu đi kèm một phantom."""
    name: str
    fov_cm: float                       # trường nhìn vật lý (cm) -> kích thước pixel
    window: Tuple[float, float]         # (level, width) mặc định để hiển thị / tính SSIM
    rois: Dict[str, Tuple[float, float, float]]  # tên -> (x, y, r) trong toạ độ chuẩn hoá
    description: str = ""
    scan_length_cm: float = 15.0        # chiều dài quét (cm) để ước tính DLP
    k_eff_dose: float = 0.0021          # hệ số chuyển DLP -> liều hiệu dụng (mSv / (mGy.cm))
    ctdi_per_mas_120: float = 0.225     # CTDIvol (mGy) trên mỗi mAs tại 120 kVp (đầu ~0.225, thân ~0.09)


# ----------------------------------------------------------------------------------------
# Công cụ dựng ellipse
# ----------------------------------------------------------------------------------------
def _grid(n: int, ss: int):
    """Lưới toạ độ chuẩn hoá [-1, 1] kích thước (n*ss)^2 (tâm pixel). y hướng LÊN (như ảnh y-up)."""
    m = n * ss
    c = (np.arange(m) + 0.5) / m * 2.0 - 1.0
    x = c[None, :].repeat(m, axis=0)
    y = -c[:, None].repeat(m, axis=1)  # hàng 0 = đỉnh ảnh = y dương
    return x, y


def _ellipse_mask(x, y, e: Ellipse):
    _, a, b, x0, y0, phi = e
    p = np.deg2rad(phi)
    cp, sp = np.cos(p), np.sin(p)
    xr = (x - x0) * cp + (y - y0) * sp
    yr = -(x - x0) * sp + (y - y0) * cp
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def _downsample(img: np.ndarray, ss: int) -> np.ndarray:
    if ss == 1:
        return img
    m = img.shape[0] // ss
    return img.reshape(m, ss, m, ss).mean(axis=(1, 3))


def render_replace(n: int, ellipses: List[Ellipse], background: float, ss: int = 4) -> np.ndarray:
    """Dựng ảnh kiểu THAY THẾ: ellipse sau ghi đè ellipse trước (phù hợp phantom HU)."""
    x, y = _grid(n, ss)
    img = np.full(x.shape, background, dtype=np.float64)
    for e in ellipses:
        img[_ellipse_mask(x, y, e)] = e[0]
    return _downsample(img, ss)


def render_additive(n: int, ellipses: List[Ellipse], ss: int = 4) -> np.ndarray:
    """Dựng ảnh kiểu CỘNG DỒN (Shepp-Logan chuẩn)."""
    x, y = _grid(n, ss)
    img = np.zeros(x.shape, dtype=np.float64)
    for e in ellipses:
        img[_ellipse_mask(x, y, e)] += e[0]
    return _downsample(img, ss)


# ----------------------------------------------------------------------------------------
# 1) Shepp-Logan (bản "modified" của Toft) -> quy đổi sang HU
# ----------------------------------------------------------------------------------------
_SL = [  # A, a, b, x0, y0, phi
    (1.0, 0.69, 0.92, 0.0, 0.0, 0),
    (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0),
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18),
    (-0.2, 0.16, 0.41, -0.22, 0.0, 18),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0),
    (0.1, 0.046, 0.046, 0.0, 0.1, 0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0),
    (0.1, 0.046, 0.023, -0.08, -0.605, 0),
    (0.1, 0.023, 0.023, 0.0, -0.606, 0),
    (0.1, 0.023, 0.046, 0.06, -0.605, 0),
]

# Bảng quy đổi giá trị Shepp-Logan -> HU:  0 -> 0 (dịch/ nước), 0.2 -> 30 (chất xám), 0.3 -> 60, 1.0 -> 1000 (xương sọ)
_SL_V = np.array([0.0, 0.2, 0.3, 0.4, 1.0])
_SL_HU = np.array([0.0, 30.0, 60.0, 90.0, 1000.0])


def shepp_logan_hu(n: int, ss: int = 4):
    v = render_additive(n, _SL, ss)
    outer = render_replace(n, [(1.0, 0.69, 0.92, 0.0, 0.0, 0)], 0.0, ss)  # mặt nạ cơ thể (0..1)
    hu_in = np.interp(v, _SL_V, _SL_HU)
    hu = outer * hu_in + (1.0 - outer) * HU_AIR
    # vùng 'v=0' bên trong hộp sọ (não thất) là DỊCH: ~ 5 HU thay vì 0 -> giữ nguyên 0 HU (nước)
    info = PhantomInfo(
        name="Shepp-Logan",
        fov_cm=25.0,
        window=(40.0, 400.0),
        rois={"bg": (0.0, -0.30, 0.12), "lesion": (0.0, 0.35, 0.08)},
        description="Shepp-Logan cải tiến (Toft) quy đổi sang HU: xương sọ ~1000 HU, chất xám ~30 HU, tổn thương ~60 HU.",
    )
    return hu, info


# ----------------------------------------------------------------------------------------
# 2) Phantom đầu (HU) - dựa trên giá trị lâm sàng (xem 'NDung 2_1_.docx': bảng HU)
# ----------------------------------------------------------------------------------------
def head_hu(n: int, ss: int = 4):
    ell: List[Ellipse] = [
        (40.0, 0.80, 0.95, 0.0, 0.0, 0),        # da / mô mềm da đầu
        (900.0, 0.73, 0.90, 0.0, 0.0, 0),       # xương sọ (cortical bone)
        (35.0, 0.665, 0.835, 0.0, 0.0, 0),      # chất xám (GM ~ 35-45 HU)
        (25.0, 0.50, 0.66, 0.0, -0.02, 0),      # chất trắng (WM ~ 20-30 HU)
        (10.0, 0.065, 0.20, -0.13, 0.12, 10),   # não thất bên trái (CSF ~ 10-15 HU)
        (10.0, 0.065, 0.20, 0.13, 0.12, -10),   # não thất bên phải
        (80.0, 0.12, 0.085, 0.40, -0.35, 30),   # khối máu tụ cấp (75-100 HU)
        (350.0, 0.025, 0.025, -0.02, -0.46, 0), # vôi hoá (tuyến tùng) ~350 HU
        (3.0, 0.09, 0.09, -0.35, 0.35, 0),      # vùng giảm tỷ trọng / phù não (tương phản thấp -22 HU)
    ]
    hu = render_replace(n, ell, HU_AIR, ss)
    info = PhantomInfo(
        name="Đầu (HU)",
        fov_cm=25.0,
        window=(40.0, 120.0),
        rois={"bg": (0.0, -0.22, 0.10), "lesion": (0.40, -0.35, 0.05)},
        description="Phantom đầu với hộp sọ, chất xám/trắng, não thất, máu tụ cấp, vôi hoá, vùng phù (tương phản thấp).",
        scan_length_cm=15.0,
        k_eff_dose=0.0021,
    )
    return hu, info


# ----------------------------------------------------------------------------------------
# 3) Phantom lồng ngực (HU): phổi, tim, cột sống, xương sườn, nốt phổi
# ----------------------------------------------------------------------------------------
def chest_hu(n: int, ss: int = 4):
    ell: List[Ellipse] = [
        (-100.0, 0.94, 0.66, 0.0, 0.0, 0),      # lớp mỡ dưới da
        (45.0, 0.89, 0.61, 0.0, 0.0, 0),        # cơ / mô mềm thành ngực
        (-760.0, 0.31, 0.43, -0.40, 0.04, -8),  # phổi trái (-700 .. -800 HU)
        (-760.0, 0.31, 0.43, 0.40, 0.04, 8),    # phổi phải
        (45.0, 0.20, 0.19, -0.05, 0.02, 20),    # tim (máu ~ 45 HU)
        (150.0, 0.045, 0.045, 0.03, 0.11, 0),   # động mạch chủ (có thuốc cản quang ~150 HU)
        (350.0, 0.075, 0.075, 0.0, -0.45, 0),   # thân đốt sống (xương xốp ~350 HU)
        (900.0, 0.02, 0.05, 0.0, -0.33, 0),     # cung sau (xương đặc)
        (40.0, 0.055, 0.055, -0.43, 0.14, 0),   # nốt phổi đặc (soft nodule ~40 HU)
        (-420.0, 0.045, 0.045, 0.45, -0.12, 0), # nốt kính mờ (ground-glass ~ -420 HU)
    ]
    # xương sườn: các ellipse nhỏ dọc theo viền ngực
    for t in np.linspace(0.45, 2.7 * np.pi / 2.0 + 0.9, 9):
        xr, yr = 0.84 * np.cos(t), 0.57 * np.sin(t)
        ell.append((750.0, 0.035, 0.022, xr, yr, np.rad2deg(t) + 90))
    hu = render_replace(n, ell, HU_AIR, ss)
    info = PhantomInfo(
        name="Lồng ngực (HU)",
        fov_cm=36.0,
        window=(-600.0, 1500.0),
        rois={"bg": (-0.05, 0.02, 0.07), "lesion": (0.03, 0.11, 0.02)},
        description="Phantom ngực: phổi (-760 HU), tim, động mạch chủ có cản quang, đốt sống, xương sườn, nốt phổi.",
        scan_length_cm=30.0,
        k_eff_dose=0.014,
        ctdi_per_mas_120=0.09,
    )
    return hu, info


# ----------------------------------------------------------------------------------------
# Phantom ngẫu nhiên (dùng huấn luyện DL - KHÔNG trùng với 3 phantom trên)
# ----------------------------------------------------------------------------------------
def random_phantom(n: int, rng: np.random.Generator, ss: int = 2):
    """Phantom ngẫu nhiên: thân (mô mềm) + vỏ xương/ khoang khí ngẫu nhiên + các ellipse mô."""
    ell: List[Ellipse] = []
    a0, b0 = rng.uniform(0.62, 0.92), rng.uniform(0.62, 0.95)
    ell.append((rng.uniform(-100, 60), a0, b0, 0.0, 0.0, rng.uniform(-20, 20)))
    if rng.random() < 0.6:  # vỏ xương
        t = rng.uniform(0.04, 0.09)
        ell.append((rng.uniform(500, 1100), a0 * (1 - t), b0 * (1 - t), 0.0, 0.0, ell[0][5]))
        ell.append((rng.uniform(10, 50), a0 * (1 - 2.2 * t), b0 * (1 - 2.2 * t), 0.0, 0.0, ell[0][5]))
    base_hu = ell[-1][0] if len(ell) > 1 else ell[0][0]
    for _ in range(rng.integers(3, 9)):
        a = rng.uniform(0.04, 0.28)
        b = rng.uniform(0.04, 0.28)
        r = rng.uniform(0.0, 0.5)
        th = rng.uniform(0, 2 * np.pi)
        kind = rng.random()
        if kind < 0.18:
            hu = rng.uniform(-850, -400)      # khí / phổi
        elif kind < 0.30:
            hu = rng.uniform(150, 700)        # xương / vôi
        elif kind < 0.40:
            hu = rng.uniform(-20, 20)         # dịch
        else:
            hu = base_hu + rng.uniform(-40, 90)  # tương phản thấp trong mô mềm
        ell.append((hu, a, b, r * np.cos(th), r * np.sin(th) * 0.9, rng.uniform(0, 180)))
    hu = render_replace(n, ell, HU_AIR, ss)
    return hu


# ----------------------------------------------------------------------------------------
# Cấy kim loại (implant)
# ----------------------------------------------------------------------------------------
def add_metal(hu: np.ndarray, n_implants: int = 2, size: float = 0.055, ss: int = 4) -> np.ndarray:
    """Thêm các implant kim loại (HU rất cao) -> gây nhiễu kim loại (photon starvation + beam hardening)."""
    n = hu.shape[0]
    spots = [(-0.40, -0.10), (0.38, -0.12), (0.0, 0.50), (0.0, -0.55)][: max(0, n_implants)]
    out = hu.copy()
    ell = [(HU_METAL, size, size, x, y, 0) for (x, y) in spots]
    if not ell:
        return out
    m = render_replace(n, [(1.0, size, size, x, y, 0) for (x, y) in spots], 0.0, ss)
    out = out * (1 - m) + HU_METAL * m
    return out


PHANTOMS = {
    "Đầu (HU)": head_hu,
    "Shepp-Logan": shepp_logan_hu,
    "Lồng ngực (HU)": chest_hu,
}

# Tích hợp các mẫu Ảnh CT Lâm Sàng Thực Tế (Real Clinical CT) và hỗ trợ nạp file tùy chỉnh
_CUSTOM_REAL_CT: Dict[str, Tuple[str, str]] = {}  # name -> (filepath, preset_name)


def register_custom_real_ct(name: str, filepath: str, preset_name: str = "Auto (Tự động)") -> str:
    """Đăng ký một ảnh CT thực tế từ đường dẫn file bất kỳ vào danh mục PHANTOMS."""
    _CUSTOM_REAL_CT[name] = (filepath, preset_name)
    if name not in PHANTOMS:
        PHANTOMS[name] = lambda n, ss=4, _n=name: _load_registered_real_ct(_n, n)
    return name


def _load_registered_real_ct(name: str, n: int):
    from .real_ct import load_real_ct_image, ensure_builtin_real_ct_samples
    if name in _CUSTOM_REAL_CT:
        fp, preset = _CUSTOM_REAL_CT[name]
        return load_real_ct_image(fp, target_n=n, preset_name=preset)
    samples = ensure_builtin_real_ct_samples()
    preset_map = {
        "CT Thực Tế: Gan Bình Thường": "Abdomen (Gan / Bụng)",
        "CT Thực Tế: Gan Nhiễm Mỡ": "Abdomen (Gan / Bụng)",
        "CT Thực Tế: Xuất Huyết Não": "Brain (Sọ não)",
        "CT Thực Tế: Phổi HRCT": "Lung (Phổi HRCT)",
    }
    return load_real_ct_image(samples[name], target_n=n, preset_name=preset_map.get(name, "Auto (Tự động)"))


for _real_name in (
    "CT Thực Tế: Xuất Huyết Não",
    "CT Thực Tế: Gan Bình Thường",
    "CT Thực Tế: Gan Nhiễm Mỡ",
    "CT Thực Tế: Phổi HRCT",
):
    PHANTOMS[_real_name] = lambda n, ss=4, _rn=_real_name: _load_registered_real_ct(_rn, n)


def make_phantom(name: str, n: int, metal: int = 0, ss: int = 4):
    """Tạo phantom hoặc nạp ảnh CT thực tế theo tên. Trả về (HU[n, n], PhantomInfo)."""
    if name not in PHANTOMS:
        raise KeyError(f"Phantom không tồn tại: {name}. Có: {list(PHANTOMS)}")
    hu, info = PHANTOMS[name](n, ss)
    if metal > 0:
        hu = add_metal(hu, metal, ss=ss)
    return hu, info


def roi_mask(n: int, roi: Tuple[float, float, float]) -> np.ndarray:
    """Mặt nạ ROI hình tròn trên lưới n x n (toạ độ chuẩn hoá)."""
    x, y = _grid(n, 1)
    cx, cy, r = roi
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def circle_mask(n: int, radius: float = 1.0) -> np.ndarray:
    x, y = _grid(n, 1)
    return x ** 2 + y ** 2 <= radius ** 2
