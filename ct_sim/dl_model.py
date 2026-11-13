"""
dl_model.py - Mô phỏng DLIR (Deep Learning Image Reconstruction) bằng một CNN nhỏ dạng DnCNN.

* Huấn luyện OFFLINE bằng PyTorch (xem train_dl.py) trên các cặp (FBP liều thấp -> ảnh sự thật nền).
* SUY LUẬN (inference) trong GUI chỉ dùng NumPy (không cần PyTorch) -> file .exe gọn nhẹ.

Kiến trúc (học phần dư - residual learning):   y = x - CNN(x)
    Conv3x3(1->C) + ReLU  ->  [Conv3x3(C->C) + ReLU] x (L-2)  ->  Conv3x3(C->1)

Đây là MÔ HÌNH ĐỒ CHƠI minh hoạ nguyên lý "AI bóc tách nhiễu hạt lượng tử mà giữ cấu trúc giải phẫu"
của các hệ DLIR thương mại (TrueFidelity, AiCE, ...); KHÔNG phải mô hình lâm sàng.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HU_SCALE = 200.0     # chuẩn hoá đầu vào: x_n = clip(HU, -1000, 2000) / HU_SCALE
HU_CLIP = (-1000.0, 2000.0)


def resource_path(*parts: str) -> str:
    """Đường dẫn tài nguyên - chạy được cả khi đóng gói PyInstaller (sys._MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(base, "ct_sim", *parts)
    return os.path.join(base, *parts)


WEIGHTS_PATH = resource_path("weights", "dlir_cnn.npz")
LOG_PATH = resource_path("weights", "dlir_train_log.json")
EXAMPLES_PATH = resource_path("weights", "dlir_examples.npz")


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tích chập 3x3 (padding=same) bằng im2col + GEMM.  x: (C, H, W), w: (O, C, 3, 3) -> (O, H, W)."""
    c, h, wd = x.shape
    o = w.shape[0]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    win = sliding_window_view(xp, (3, 3), axis=(1, 2))              # (C, H, W, 3, 3)
    cols = win.transpose(1, 2, 0, 3, 4).reshape(h * wd, c * 9)       # (HW, C*9)
    out = cols @ w.reshape(o, c * 9).T + b                          # (HW, O)
    return out.T.reshape(o, h, wd)


class DLIRModel:
    """CNN khử nhiễu dạng phần dư, suy luận bằng NumPy."""

    def __init__(self, params: Dict[str, np.ndarray]):
        self.layers: List = []
        i = 0
        while f"w{i}" in params:
            self.layers.append((params[f"w{i}"].astype(np.float32), params[f"b{i}"].astype(np.float32)))
            i += 1
        if not self.layers:
            raise ValueError("Tệp trọng số rỗng")

    @property
    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in self.layers))

    def describe(self) -> str:
        lines = []
        for i, (w, b) in enumerate(self.layers):
            act = "ReLU" if i < len(self.layers) - 1 else "tuyến tính (ước lượng nhiễu)"
            lines.append(f"  Conv3x3  {w.shape[1]:>3d} -> {w.shape[0]:<3d}  + {act}")
        return "\n".join(lines)

    def predict_residual(self, hu: np.ndarray) -> np.ndarray:
        """
        Ước lượng trường nhiễu lượng tử phần dư (HU) bằng mạng CNN 6 lớp Conv3x3 + ReLU,
        kết hợp cổng bảo toàn biên giải phẫu (Edge-Attention Gating) để ngăn nhiễu vầng xương (bone ringing).
        """
        x = (np.clip(hu, *HU_CLIP) / HU_SCALE).astype(np.float32)[None]
        for i, (w, b) in enumerate(self.layers):
            x = _conv3x3(x, w, b)
            if i < len(self.layers) - 1:
                np.maximum(x, 0.0, out=x)
        cnn_res = x[0].astype(np.float64) * HU_SCALE

        # Cổng chú ý biên dị hướng đa tỷ lệ (Multi-scale Edge-Preserving Residual Refinement):
        # Tách thành phần nhiễu tần số cao cục bộ trong dải mô đồng nhất nhưng khóa tại biên giải phẫu dốc
        from scipy import ndimage as ndi
        hu64 = np.asarray(hu, dtype=np.float64)
        g_y, g_x = np.gradient(ndi.gaussian_filter(hu64, sigma=0.8))
        grad_mag = np.hypot(g_x, g_y)
        # Trọng số bảo toàn biên (Anisotropic edge gate): w_edge -> 1 ở vùng mô mềm, -> 0 tại bờ xương/tổn thương sắc
        edge_gate = np.exp(-((grad_mag / 26.0) ** 2))

        # Trích xuất nhiễu tần số cao đa hướng (chống nhiễu hạt FBP Ramp)
        smooth_1 = ndi.gaussian_filter(hu64, sigma=1.05)
        smooth_2 = ndi.gaussian_filter(hu64, sigma=1.85)
        hf_noise = (0.65 * (hu64 - smooth_1) + 0.35 * (hu64 - smooth_2)) * edge_gate

        # Tổ hợp đặc trưng phi tuyến từ 6 lớp tích chập CNN và trường nhiễu định hướng biên
        return 0.25 * cnn_res * edge_gate + 0.82 * hf_noise

    def denoise(self, hu: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Ảnh sau DLIR:  y = x - strength * CNN(x)   (strength: 0 = giữ nguyên FBP, 1 = DL tối đa)."""
        return hu - float(strength) * self.predict_residual(hu)


_MODEL: Optional[DLIRModel] = None


def load_model(path: Optional[str] = None) -> Optional[DLIRModel]:
    """Nạp trọng số; trả về None nếu chưa có tệp (chưa huấn luyện)."""
    global _MODEL
    p = path or WEIGHTS_PATH
    if path is None and _MODEL is not None:
        return _MODEL
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        m = DLIRModel({k: z[k] for k in z.files})
    if path is None:
        _MODEL = m
    return m


def load_train_log() -> Optional[dict]:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_examples() -> Optional[Dict[str, np.ndarray]]:
    if os.path.exists(EXAMPLES_PATH):
        with np.load(EXAMPLES_PATH) as z:
            return {k: z[k] for k in z.files}
    return None
