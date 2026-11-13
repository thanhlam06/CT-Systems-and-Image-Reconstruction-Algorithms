"""
metrics.py - Các chỉ số chất lượng ảnh CT: RMSE, PSNR, SSIM (trong cửa sổ hiển thị, vùng mô mềm), nhiễu (SD trong ROI
đồng nhất), độ lệch HU, độ tương phản và CNR.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

from .phantoms import PhantomInfo, circle_mask, roi_mask


def body_mask(truth_hu: np.ndarray) -> np.ndarray:
    """Mặt nạ vùng cơ thể (loại không khí ngoài)."""
    n = truth_hu.shape[0]
    return (truth_hu > -950.0) & circle_mask(n, 0.98)


def soft_mask(truth_hu: np.ndarray, info: PhantomInfo) -> np.ndarray:
    """Vùng đánh giá 'mô mềm': thuộc cơ thể, nằm trong cửa sổ hiển thị và cách xa biên xương/khí mạnh (>150 HU/pixel)."""
    lvl, wid = info.window
    gy, gx = np.gradient(truth_hu)
    strong = np.hypot(gx, gy) > 150.0
    strong = ndi.binary_dilation(strong, iterations=2)
    inwin = np.abs(truth_hu - lvl) <= wid / 2.0 + 100.0
    return body_mask(truth_hu) & inwin & ~strong


def _window(img: np.ndarray, level: float, width: float) -> np.ndarray:
    lo, hi = level - width / 2.0, level + width / 2.0
    return (np.clip(img, lo, hi) - lo) / (hi - lo)


def evaluate(img_hu: np.ndarray, truth_hu: np.ndarray, info: PhantomInfo) -> Dict[str, float]:
    """Trả về từ điển chỉ số cho ảnh `img_hu` so với `truth_hu` (cùng kích thước).

    * RMSE_HU   : sai số bình phương trung bình toàn cơ thể (HU)
    * PSNR_dB   : PSNR trong cửa sổ hiển thị (đỉnh = bề rộng cửa sổ), tính trên vùng mô mềm
    * SSIM      : SSIM trong cửa sổ hiển thị, trung bình trên vùng mô mềm
    * Noise_HU  : độ lệch chuẩn trong ROI đồng nhất (nền)
    * CNR       : |trung bình tổn thương - nền| / nhiễu
    """
    n = truth_hu.shape[0]
    bm = body_mask(truth_hu)
    sm = soft_mask(truth_hu, info)
    lvl, wid = info.window
    a, b = _window(img_hu, lvl, wid), _window(truth_hu, lvl, wid)

    err = img_hu - truth_hu
    rmse_full = float(np.sqrt(np.mean(err[bm] ** 2)))
    mse_w = float(np.mean(((a - b) ** 2)[sm]))
    psnr = float(10.0 * np.log10(1.0 / max(mse_w, 1e-12)))
    _, ssim_map = structural_similarity(a, b, data_range=1.0, full=True)
    ssim = float(np.mean(ssim_map[sm]))

    bg = roi_mask(n, info.rois["bg"])
    le = roi_mask(n, info.rois["lesion"])
    noise = float(np.std(img_hu[bg]))
    bias = float(np.mean(img_hu[bg]) - np.mean(truth_hu[bg]))
    contrast = float(np.mean(img_hu[le]) - np.mean(img_hu[bg]))
    cnr = float(abs(contrast) / max(noise, 1e-6))
    return {
        "RMSE_HU": rmse_full,
        "PSNR_dB": psnr,
        "SSIM": ssim,
        "Noise_HU": noise,
        "Bias_HU": bias,
        "Contrast_HU": contrast,
        "CNR": cnr,
    }
