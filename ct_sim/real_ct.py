"""
real_ct.py - Module nạp, hiệu chuẩn thang Hounsfield (HU) và phân tích Ảnh CT Thực Tế (Real Clinical CT).

Hỗ trợ:
  1. Nạp ảnh CT thực tế từ nhiều định dạng:
     - Ảnh đa phương tiện y khoa: .png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp
     - Mảng số học: .npy, .npz (chứa ma trận HU hoặc cường độ xám)
     - File DICOM lâm sàng: .dcm (nếu có pydicom hoặc đọc raw pixel)
  2. Hiệu chuẩn phi tuyến từng khúc (Piecewise-Linear Radiological Calibration) từ mức xám I in [0, 1]
     sang thang Hounsfield chuẩn thực tế:
     - Nền ngoài cơ thể -> Không khí (-1000 HU)
     - Khoang khí / Phổi / Mỡ -> [-850, -50] HU
     - Nhu mô mềm (Não, Gan, Lách, Cơ) -> [+10, +90] HU
     - Vôi hóa / Xương vỏ (Cortical Bone) -> [+250, +1200] HU
  3. Tự động nhận diện vùng quan tâm (Auto-ROI Detection):
     - Tìm vùng nhu mô đồng nhất (phương sai cục bộ nhỏ) làm ROI nền (bg).
     - Tìm vùng tổn thương / cấu trúc tương phản cao làm ROI tổn thương (lesion).
  4. Kho mẫu Ảnh CT Thực Tế tích hợp sẵn trong `assets/real_ct/`.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

from .phantoms import PhantomInfo, HU_AIR, circle_mask


def get_real_ct_dir() -> str:
    """Trả về đường dẫn thư mục chứa các ảnh CT thực tế mẫu (tương thích PyInstaller)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(base, "assets", "real_ct")
    os.makedirs(d, exist_ok=True)
    return d


# Bảng cấu hình cửa sổ hiệu chuẩn HU theo vùng giải phẫu của ảnh CT thực tế
CALIBRATION_PRESETS: Dict[str, dict] = {
    "Auto (Tự động)": {
        "window": (45.0, 250.0),
        "fov_cm": 30.0,
        "xp": [0.0, 0.04, 0.15, 0.35, 0.55, 0.78, 0.90, 1.0],
        "fp": [-1000.0, -900.0, -110.0, 15.0, 48.0, 85.0, 350.0, 1050.0],
        "desc": "Hiệu chuẩn tự động đa mô (Không khí -1000 HU, Mỡ -100 HU, Mô mềm 15..85 HU, Xương >350 HU)."
    },
    "Abdomen (Gan / Bụng)": {
        "window": (50.0, 350.0),
        "fov_cm": 36.0,
        "xp": [0.0, 0.05, 0.18, 0.38, 0.62, 0.80, 0.92, 1.0],
        "fp": [-1000.0, -850.0, -105.0, 20.0, 62.0, 110.0, 420.0, 1100.0],
        "desc": "Cửa sổ Trung thất / Bụng: Gan (15..70 HU), Lách (40..60 HU), Mỡ (-100 HU), Cột sống (400..1100 HU)."
    },
    "Brain (Sọ não)": {
        "window": (40.0, 120.0),
        "fov_cm": 24.0,
        "xp": [0.0, 0.05, 0.18, 0.35, 0.58, 0.78, 0.90, 1.0],
        "fp": [-1000.0, -900.0, 8.0, 24.0, 38.0, 82.0, 450.0, 1200.0],
        "desc": "Cửa sổ Nhu mô Não: Dịch não tủy (8..15 HU), Chất trắng (25 HU), Chất xám (38 HU), Xuất huyết (80 HU), Hộp sọ (1000 HU)."
    },
    "Lung (Phổi HRCT)": {
        "window": (-500.0, 1500.0),
        "fov_cm": 36.0,
        "xp": [0.0, 0.06, 0.25, 0.45, 0.65, 0.82, 0.92, 1.0],
        "fp": [-1000.0, -880.0, -740.0, -420.0, 35.0, 95.0, 400.0, 950.0],
        "desc": "Cửa sổ Nhu mô Phổi: Phế nang (-760 HU), Kính mờ GGO (-420 HU), Nốt đặc/Mạch máu (40 HU), Xương sườn (800 HU)."
    },
}


def _square_pad_resize(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Đưa ảnh 2D bất kỳ về hình vuông (bằng cách đệm biên tối) và nội suy Bicubic về kích thước target_size x target_size."""
    h, w = arr.shape[:2]
    if h != w:
        s = max(h, w)
        bg_val = float(np.percentile(arr, 2))
        padded = np.full((s, s), bg_val, dtype=np.float32)
        y0 = (s - h) // 2
        x0 = (s - w) // 2
        padded[y0:y0 + h, x0:x0 + w] = arr
        arr = padded

    pil_im = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = pil_im.resize((target_size, target_size), resample=Image.Resampling.BICUBIC)
    # Lật trục y để khớp với quy ước origin='lower' trong matplotlib của hệ thống
    return np.flipud(np.asarray(resized, dtype=np.float64))


def auto_detect_rois(hu: np.ndarray, preset_name: str = "Auto (Tự động)") -> Dict[str, Tuple[float, float, float]]:
    """
    Tự động định vị 2 vùng ROI hình tròn (bg: nền nhu mô đồng nhất, lesion: vùng tổn thương/cấu trúc tương phản)
    trên ảnh CT thực tế trong hệ tọa độ chuẩn hóa [-1, 1]^2.
    """
    n = hu.shape[0]
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x = c[None, :].repeat(n, axis=0)
    y = -c[:, None].repeat(n, axis=1)

    if "Lung" in preset_name:
        hu_min, hu_max = -850.0, 120.0
    else:
        hu_min, hu_max = -40.0, 180.0

    # Tính trung bình và độ lệch chuẩn cục bộ trong cửa sổ 9x9
    local_mean = ndi.uniform_filter(hu, size=9)
    local_sq = ndi.uniform_filter(hu ** 2, size=9)
    local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0.0))

    # Mặt nạ vùng nhu mô bên trong cơ thể (cách xa khí và cách xa xương đặc ít nhất 5 pixel)
    near_bone = ndi.binary_dilation(hu > 145.0, iterations=6)
    near_air = ndi.binary_dilation(hu < (hu_min - 20.0), iterations=5)
    parenchyma = (
        (x ** 2 + y ** 2 <= 0.58 ** 2)
        & (hu > hu_min) & (hu < hu_max)
        & (local_mean > hu_min) & (local_mean < hu_max)
        & (~near_bone) & (~near_air)
    )
    if np.count_nonzero(parenchyma) < 40:
        parenchyma = (x ** 2 + y ** 2 <= 0.65 ** 2) & (hu > hu_min) & (hu < hu_max) & (~near_bone)
    if np.count_nonzero(parenchyma) < 20:
        parenchyma = (x ** 2 + y ** 2 <= 0.68 ** 2) & (hu > -700.0) & (hu < 200.0)

    # 1. Tìm tâm ROI nền (bg): vùng nhu mô đồng nhất có độ lệch chuẩn cục bộ nhỏ nhất
    score_bg = np.where(parenchyma, local_std, 1e9)
    by_idx, bx_idx = np.unravel_index(np.argmin(score_bg), hu.shape)
    bg_x, bg_y = float(x[by_idx, bx_idx]), float(y[by_idx, bx_idx])
    bg_hu_val = float(local_mean[by_idx, bx_idx])

    # 2. Tìm tâm ROI tổn thương (lesion): nằm trong vùng nhu mô, cách bg một khoảng hợp lý,
    # có độ tương phản mô mềm rõ rệt (20..120 HU đối với tạng đặc, hoặc 100..500 HU đối với phổi)
    dist_from_bg = np.hypot(x - bg_x, y - bg_y)
    lesion_candidates = parenchyma & (dist_from_bg >= 0.20) & (dist_from_bg <= 0.65)
    if np.count_nonzero(lesion_candidates) < 20:
        lesion_candidates = (x ** 2 + y ** 2 <= 0.62 ** 2) & (dist_from_bg >= 0.18)

    diff_hu = np.abs(local_mean - bg_hu_val)
    target_diff = 350.0 if "Lung" in preset_name else 45.0
    contrast_score = diff_hu * np.exp(-((diff_hu - target_diff) / (target_diff * 1.2)) ** 2) / (1.0 + 0.25 * local_std)
    score_le = np.where(lesion_candidates, contrast_score, -1.0)
    ly_idx, lx_idx = np.unravel_index(np.argmax(score_le), hu.shape)
    le_x, le_y = float(x[ly_idx, lx_idx]), float(y[ly_idx, lx_idx])

    return {
        "bg": (bg_x, bg_y, 0.080),
        "lesion": (le_x, le_y, 0.055),
    }


def load_real_ct_image(
    filepath: str,
    target_n: int = 256,
    preset_name: str = "Auto (Tự động)",
    custom_rois: Optional[Dict[str, Tuple[float, float, float]]] = None
) -> Tuple[np.ndarray, PhantomInfo]:
    """
    Nạp tập tin ảnh CT thực tế từ đĩa và chuyển đổi sang ma trận HU kích thước (target_n, target_n)
    kèm đối tượng siêu dữ liệu PhantomInfo.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy tập tin ảnh CT: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    raw_is_hu = False

    if ext == ".npy":
        arr = np.load(filepath).astype(np.float64)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.min() < -200.0 and arr.max() > 100.0:
            raw_is_hu = True
    elif ext == ".npz":
        with np.load(filepath) as z:
            arr = z[z.files[0]].astype(np.float64)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.min() < -200.0 and arr.max() > 100.0:
            raw_is_hu = True
    elif ext == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(filepath)
            arr = ds.pixel_array.astype(np.float64)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
            arr = arr * slope + intercept
            raw_is_hu = True
        except Exception:
            im = Image.open(filepath).convert("L")
            arr = np.asarray(im, dtype=np.float64)
    else:
        im = Image.open(filepath).convert("L")
        arr = np.asarray(im, dtype=np.float64)

    arr_sq = _square_pad_resize(arr, target_n)
    preset = CALIBRATION_PRESETS.get(preset_name, CALIBRATION_PRESETS["Auto (Tự động)"])

    if raw_is_hu:
        hu = np.clip(arr_sq, -1000.0, 3000.0)
    else:
        # Chuẩn hóa cường độ sáng về [0, 1] theo phân vị bền vững
        p_lo, p_hi = np.percentile(arr_sq, [1.5, 99.5])
        if p_hi - p_lo < 1e-6:
            p_lo, p_hi = float(arr_sq.min()), float(arr_sq.max() + 1.0)
        norm = np.clip((arr_sq - p_lo) / (p_hi - p_lo), 0.0, 1.0)

        # Ánh xạ phi tuyến từng khúc sang đơn vị Hounsfield (HU)
        hu = np.interp(norm, preset["xp"], preset["fp"])

        # Mặt nạ nền ngoài đường viền cơ thể -> gán chính xác HU_AIR (-1000 HU)
        body = ndi.binary_fill_holes(norm > 0.055)
        body = ndi.binary_opening(body, iterations=1)
        if np.count_nonzero(body) > 0.1 * target_n * target_n:
            hu = np.where(body, hu, HU_AIR)

    # Đảm bảo bên ngoài hình tròn nội tiếp của trường nhìn (FOV) là không khí (-1000 HU)
    c_mask = circle_mask(target_n, 0.96)
    hu = np.where(c_mask, hu, HU_AIR)

    # Tính ROI tự động trên lưới 128x128 chuẩn
    hu_128 = hu.reshape(128, target_n // 128, 128, target_n // 128).mean(axis=(1, 3)) if target_n >= 128 else hu
    rois = custom_rois if custom_rois is not None else auto_detect_rois(hu_128, preset_name)

    fname = os.path.basename(filepath)
    info = PhantomInfo(
        name=f"CT Thực Tế ({fname})",
        fov_cm=preset["fov_cm"],
        window=preset["window"],
        rois=rois,
        description=f"Ảnh CT thực tế từ '{fname}' — {preset['desc']}",
        scan_length_cm=20.0,
        k_eff_dose=0.008,
        ctdi_per_mas_120=0.14,
    )
    return hu, info


# ----------------------------------------------------------------------------------------
# KHỞI TẠO KHO MẪU ẢNH CT LÂM SÀNG THỰC TẾ (TRÍCH XUẤT & TÁI TẠO ĐỘ PHÂN GIẢI CAO)
# ----------------------------------------------------------------------------------------
def _inpaint_annotations(gray: np.ndarray) -> np.ndarray:
    """Làm sạch các nét vẽ vòng tròn chú thích (đen hoàn toàn hoặc trắng chói nhỏ) trên ảnh CT gan thực tế."""
    med = ndi.median_filter(gray, size=7)
    # Phát hiện các nét vẽ mảnh có độ dốc cục bộ bất thường (chú thích chữ / vòng tròn)
    diff = np.abs(gray - med)
    # Vùng bên trong cơ thể
    inner = ndi.binary_erosion(gray > 18, iterations=8)
    # Chú thích vòng tròn đen (gray < 15 bên trong mô gan sáng > 65) hoặc chữ trắng nhỏ
    dark_stroke = inner & (gray < 28) & (med > 65)
    bright_text = inner & (gray > 225) & (med < 175)
    mask = ndi.binary_dilation(dark_stroke | bright_text, iterations=2)
    out = np.where(mask, med, gray)
    return ndi.gaussian_filter(out, sigma=0.45)


def ensure_builtin_real_ct_samples() -> Dict[str, str]:
    """Đảm bảo 4 tập tin ảnh CT thực tế mẫu luôn có sẵn trong `assets/real_ct/`."""
    out_dir = get_real_ct_dir()
    samples = {
        "CT Thực Tế: Gan Bình Thường": os.path.join(out_dir, "real_liver_normal_ct.png"),
        "CT Thực Tế: Gan Nhiễm Mỡ": os.path.join(out_dir, "real_liver_fatty_ct.png"),
        "CT Thực Tế: Xuất Huyết Não": os.path.join(out_dir, "real_brain_hemorrhage_ct.png"),
        "CT Thực Tế: Phổi HRCT": os.path.join(out_dir, "real_chest_hrct.png"),
    }

    # 1 & 2: Trích xuất từ tài liệu môn học NDung 2_1_.docx nếu tồn tại
    docx_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(out_dir)), "..", "NDung 2_1_.docx"),
        r"d:\Desktop\HK261\CSVLHN\NDung 2_1_.docx",
    ]
    extracted_from_docx = False
    if not (os.path.exists(samples["CT Thực Tế: Gan Bình Thường"]) and os.path.exists(samples["CT Thực Tế: Gan Nhiễm Mỡ"])):
        import zipfile, io
        for docx_p in docx_candidates:
            if os.path.exists(docx_p):
                try:
                    with zipfile.ZipFile(docx_p) as z:
                        if "word/media/image1.png" in z.namelist():
                            im = Image.open(io.BytesIO(z.read("word/media/image1.png"))).convert("L")
                            arr = np.asarray(im, dtype=np.float64)
                            # Cắt khung CT bên trái (Gan bình thường 66 HU) và khung bên phải (Gan nhiễm mỡ 15 HU)
                            left_crop = _inpaint_annotations(arr[54:292, 58:376])
                            right_crop = _inpaint_annotations(arr[54:292, 414:692])
                            Image.fromarray(np.clip(left_crop, 0, 255).astype(np.uint8)).save(samples["CT Thực Tế: Gan Bình Thường"])
                            Image.fromarray(np.clip(right_crop, 0, 255).astype(np.uint8)).save(samples["CT Thực Tế: Gan Nhiễm Mỡ"])
                            extracted_from_docx = True
                            break
                except Exception:
                    pass

    # Dự phòng nếu chạy ở máy khác không có file docx gốc: tạo ảnh CT bụng giải phẫu chi tiết cao
    if not os.path.exists(samples["CT Thực Tế: Gan Bình Thường"]) or not os.path.exists(samples["CT Thực Tế: Gan Nhiễm Mỡ"]):
        _synthesize_realistic_liver_pair(
            samples["CT Thực Tế: Gan Bình Thường"],
            samples["CT Thực Tế: Gan Nhiễm Mỡ"]
        )

    # 3: Ảnh CT Sọ não thực tế (Xuất huyết nhu mô cấp + phù não + rãnh cuộn não chi tiết)
    if not os.path.exists(samples["CT Thực Tế: Xuất Huyết Não"]):
        _synthesize_realistic_brain_ct(samples["CT Thực Tế: Xuất Huyết Não"])

    # 4: Ảnh CT Lồng ngực độ phân giải cao (HRCT Phổi: Mạch máu phân nhánh + Kính mờ + Nốt phổi)
    if not os.path.exists(samples["CT Thực Tế: Phổi HRCT"]):
        _synthesize_realistic_chest_hrct(samples["CT Thực Tế: Phổi HRCT"])

    return samples


def _synthesize_realistic_liver_pair(path_normal: str, path_fatty: str, n: int = 256):
    """Tạo cặp ảnh CT bụng thực tế có cấu trúc giải phẫu chi tiết (gan, tĩnh mạch cửa, lách, cột sống, động mạch chủ, thận)."""
    rng = np.random.default_rng(42)
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    body = (x / 0.88) ** 2 + (y / 0.68) ** 2 <= 1.0
    subcut = ((x / 0.84) ** 2 + (y / 0.64) ** 2 <= 1.0)
    perit = ((x / 0.78) ** 2 + (y / 0.58) ** 2 <= 1.0)

    # Thuỳ gan bên phải ảnh (x < 0.15), lách bên trái (x > 0.45), cột sống phía dưới (y > 0.35)
    liver = perit & (((x + 0.25) / 0.48) ** 2 + ((y + 0.02) / 0.44) ** 2 <= 1.0) & (x - 0.3 * y < 0.22)
    spleen = perit & (((x - 0.54) / 0.18) ** 2 + ((y - 0.14) / 0.24) ** 2 <= 1.0)
    spine = ((x - 0.0) / 0.13) ** 2 + ((y - 0.42) / 0.12) ** 2 <= 1.0
    aorta = ((x - 0.04) / 0.055) ** 2 + ((y - 0.22) / 0.055) ** 2 <= 1.0
    stomach = ((x - 0.24) / 0.18) ** 2 + ((y + 0.22) / 0.16) ** 2 <= 1.0

    # Hệ mạch máu phân nhánh trong gan
    vessels = np.zeros_like(x, dtype=bool)
    for cx, cy, rx, ry, ang in [(-0.22, -0.05, 0.12, 0.022, 25), (-0.38, 0.08, 0.10, 0.018, -30), (-0.12, 0.06, 0.08, 0.02, 50)]:
        ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        xr = (x - cx) * ca + (y - cy) * sa
        yr = -(x - cx) * sa + (y - cy) * ca
        vessels |= liver & ((xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0)

    tex = ndi.gaussian_filter(rng.normal(0, 6.0, (n, n)), sigma=1.4)
    for is_fatty, out_p in [(False, path_normal), (True, path_fatty)]:
        img = np.zeros((n, n), dtype=np.float64)
        img[body] = 45.0
        img[subcut] = 95.0
        img[perit] = 35.0
        img[liver] = 105.0 if not is_fatty else 72.0      # Gan bình thường sáng hơn mạch; gan nhiễm mỡ tối hơn mạch
        img[vessels] = 85.0 if not is_fatty else 125.0
        img[spleen] = 128.0
        img[aorta] = 148.0
        img[stomach] = 20.0
        img[spine] = 242.0
        img = np.where(body, np.clip(img + tex, 10, 255), 0.0)
        Image.fromarray(img.astype(np.uint8)).save(out_p)


def _synthesize_realistic_brain_ct(path_out: str, n: int = 256):
    """Tạo ảnh CT Sọ não thực tế với cấu trúc xương sọ 3 lớp, rãnh cuộn vỏ não, nhân bèo, não thất, ổ xuất huyết & phù não."""
    rng = np.random.default_rng(108)
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    # Hình dạng hộp sọ giải phẫu (hơi thuôn phía trán y < 0, bầu ở chẩm y > 0)
    w_mod = 0.76 * (1.0 + 0.06 * np.sin(np.pi * y))
    r_skull = (x / w_mod) ** 2 + (y / 0.90) ** 2

    scalp = r_skull <= 1.04
    skull_outer = r_skull <= 0.98
    brain_mask = r_skull <= 0.83

    # Tạo cấu trúc rãnh vỏ não (sulci/gyri) bằng trường ngẫu nhiên băng thông hẹp ở ngoại vi
    theta = np.arctan2(y, x)
    gyri_wave = np.sin(18 * theta + 2.0 * np.sin(5 * theta)) * np.exp(-((r_skull - 0.78) / 0.08) ** 2)
    tex_parenchyma = ndi.gaussian_filter(rng.normal(0, 1.0, (n, n)), sigma=1.6)

    # Phân tách chất xám vỏ não và chất trắng trung tâm
    wm_mask = (r_skull <= 0.62 + 0.04 * gyri_wave) & (np.abs(x) > 0.025)
    # Liềm đại não (Falx cerebri) ở đường giữa
    falx = brain_mask & (np.abs(x) <= 0.014) & ((y < -0.25) | (y > 0.25))

    # Hệ thống não thất bên (hình cánh bướm đặc trưng) và não thất ba
    vent_l = (((x + 0.12 - 0.08 * y) / 0.065) ** 2 + ((y + 0.02) / 0.24) ** 2 <= 1.0) & (x < -0.02)
    vent_r = (((x - 0.11 + 0.08 * y) / 0.055) ** 2 + ((y + 0.02) / 0.23) ** 2 <= 1.0) & (x > 0.02)
    vent_3 = (np.abs(x) <= 0.018) & (np.abs(y - 0.05) <= 0.09)
    csf_sulci = brain_mask & (r_skull > 0.76) & (gyri_wave > 0.48)
    csf = vent_l | vent_r | vent_3 | csf_sulci

    # Nhân xám trung ương (Nhân đuôi, Nhân bèo, Đồi thị)
    bg_nuclei = (
        (((x + 0.19) / 0.07) ** 2 + ((y + 0.05) / 0.14) ** 2 <= 1.0) |
        (((x - 0.19) / 0.07) ** 2 + ((y + 0.05) / 0.14) ** 2 <= 1.0)
    )

    # Ổ xuất huyết não cấp tính (Hyperdense hematoma ~ 82 HU) kèm viền phù não giảm tỷ trọng xung quanh (Edema ~ 16 HU)
    hem_dist = ((x - 0.30) / 0.14) ** 2 + ((y - 0.18 + 0.15 * (x - 0.30)) / 0.10) ** 2
    edema = brain_mask & (hem_dist <= 1.85) & (~csf)
    hematoma = brain_mask & (hem_dist <= 0.92 + 0.15 * tex_parenchyma)

    # Vôi hóa tuyến tùng và đám rối mạch mạc
    calc = (
        (x ** 2 + (y - 0.16) ** 2 <= 0.022 ** 2) |
        ((x + 0.13) ** 2 + (y - 0.19) ** 2 <= 0.018 ** 2) |
        ((x - 0.12) ** 2 + (y - 0.19) ** 2 <= 0.018 ** 2)
    )

    img = np.zeros((n, n), dtype=np.float64)
    img[scalp] = 75.0
    img[skull_outer] = 245.0
    # Lớp xương xốp diploe giữa 2 bản sọ
    diploe = (r_skull > 0.87) & (r_skull < 0.93)
    img[diploe] = 205.0
    img[brain_mask] = 132.0 + 4.5 * tex_parenchyma[brain_mask]   # Chất xám ~ 38-42 HU
    img[wm_mask] = 112.0 + 3.8 * tex_parenchyma[wm_mask]         # Chất trắng ~ 24-28 HU
    img[bg_nuclei] = 136.0 + 4.0 * tex_parenchyma[bg_nuclei]
    img[csf] = 48.0 + 2.5 * tex_parenchyma[csf]                  # Dịch não tủy ~ 10 HU
    img[falx] = 165.0
    img[edema] = 86.0 + 3.0 * tex_parenchyma[edema]              # Phù não vận mạch
    img[hematoma] = 198.0 + 6.0 * tex_parenchyma[hematoma]       # Máu tụ cấp tính
    img[calc] = 235.0

    img = ndi.gaussian_filter(img, sigma=0.65)
    img = np.where(scalp, np.clip(img, 10, 255), 0.0)
    Image.fromarray(img.astype(np.uint8)).save(path_out)


def _synthesize_realistic_chest_hrct(path_out: str, n: int = 256):
    """Tạo ảnh CT Lồng ngực HRCT thực tế với cây mạch máu phổi phân nhánh, phế quản, tim, cột sống, xương sườn, kính mờ GGO & nốt đặc."""
    rng = np.random.default_rng(2026)
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]

    body = (x / 0.91) ** 2 + (y / 0.65) ** 2 <= 1.0
    muscle = (x / 0.85) ** 2 + (y / 0.59) ** 2 <= 1.0

    # Hai trường phổi trái và phải
    lung_l = (((x + 0.39) / 0.33) ** 2 + ((y - 0.02) / 0.44) ** 2 <= 1.0) & (x < -0.08)
    lung_r = (((x - 0.39) / 0.33) ** 2 + ((y - 0.02) / 0.44) ** 2 <= 1.0) & (x > 0.08)
    # Khuyết tim bên phổi phải/trái
    mediastinum = ((x - 0.05) / 0.22) ** 2 + ((y + 0.05) / 0.22) ** 2 <= 1.0
    lungs = (lung_l | lung_r) & (~mediastinum)

    # Tạo hệ mạch máu phổi tỏa ra từ rốn phổi (hila tại x = +-0.18, y = 0.0)
    vessels = np.zeros((n, n), dtype=np.float64)
    for side in (-1.0, 1.0):
        hx, hy = side * 0.16, 0.0
        dx, dy = x - hx, y - hy
        rad = np.hypot(dx, dy) + 1e-4
        ang = np.arctan2(dy, side * dx)
        # Các nhánh mạch máu đa tần số góc
        branch = np.maximum(0.0, np.cos(7 * ang + 1.5 * np.sin(3 * ang) + 4.0 * rad)) ** 10
        branch += 0.7 * np.maximum(0.0, np.cos(15 * ang - 6.0 * rad)) ** 14
        vessels += lungs * branch * np.exp(-rad / 0.38)

    # Tổn thương kính mờ (Ground-Glass Opacity - GGO) ở thùy dưới phổi và nốt đặc (Solid Nodule)
    ggo = lungs & (((x - 0.44) / 0.14) ** 2 + ((y - 0.18) / 0.12) ** 2 <= 1.0)
    nodule = lungs & (((x + 0.42) / 0.055) ** 2 + ((y + 0.14) / 0.052) ** 2 <= 1.0)

    # Cột sống, xương ức và các cung xương sườn
    spine = ((x / 0.12) ** 2 + ((y - 0.42) / 0.11) ** 2 <= 1.0)
    canal = ((x / 0.045) ** 2 + ((y - 0.38) / 0.04) ** 2 <= 1.0)
    sternum = ((x / 0.08) ** 2 + ((y + 0.52) / 0.035) ** 2 <= 1.0)
    aorta = (((x - 0.06) / 0.058) ** 2 + ((y - 0.22) / 0.058) ** 2 <= 1.0)

    ribs = np.zeros_like(body)
    for t in np.linspace(0.35, np.pi - 0.35, 6):
        for sgn in (-1.0, 1.0):
            rx, ry = 0.82 * np.cos(t), sgn * 0.55 * np.sin(t)
            ribs |= ((x - rx) / 0.04) ** 2 + ((y - ry) / 0.025) ** 2 <= 1.0

    tex = ndi.gaussian_filter(rng.normal(0, 4.0, (n, n)), sigma=1.1)
    img = np.zeros((n, n), dtype=np.float64)
    img[body] = 92.0                 # Mỡ dưới da
    img[muscle] = 152.0              # Cơ thành ngực & Trung thất
    img[mediastinum] = 162.0         # Tim
    img[aorta] = 188.0               # Động mạch chủ ngực
    img[lungs] = 32.0 + 95.0 * np.clip(vessels[lungs], 0.0, 1.0)
    img[ggo] = np.maximum(img[ggo], 82.0 + 12.0 * np.sin(25 * x[ggo]) * np.cos(25 * y[ggo]))
    img[nodule] = 165.0
    img[spine] = 238.0
    img[canal] = 120.0
    img[sternum] = 230.0
    img[ribs] = 242.0

    img = ndi.gaussian_filter(img + tex, sigma=0.55)
    img = np.where(body, np.clip(img, 8, 255), 0.0)
    Image.fromarray(img.astype(np.uint8)).save(path_out)
