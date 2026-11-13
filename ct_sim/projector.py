"""
projector.py - Toán tử chiếu xuôi / chiếu ngược (ma trận thưa) dùng cho các thuật toán lặp (IR) và hoạt hình FBP.

Hai mô hình:
  * 'strip'  : tích phân theo DIỆN TÍCH (area-weighted): mỗi pixel vuông có "dấu chân" hình thang trên trục đầu dò;
               trọng số a_ij = phần diện tích pixel j nằm trong dải rộng 1 bin của tia i  (chính xác hơn, ít răng cưa).
               -> dùng cho SART / SIRT / PWLS (IR).
  * 'linear' : pixel-driven + nội suy tuyến tính (giống skimage.iradon)  -> dùng cho FBP / hoạt hình.

Quy ước hình học giống skimage.transform.radon:   s = x*cos(theta) + y*sin(theta)   (x sang phải, y hướng lên).

A  : (n_views * n_det) x (N*N)    p = A @ x       (chiếu xuôi,  'Radon rời rạc')
A^T: (N*N) x (n_views * n_det)    b = A.T @ p     (chiếu ngược - toán tử LIÊN HỢP chính xác)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import scipy.sparse as sp


def _trap_cdf(s, L, M, ab, mx):
    """CDF của dấu chân hình thang (tích chập hai hộp rộng a, b): hỗ trợ [-L, L], phẳng trên [-M, M]."""
    s = np.asarray(s)
    rise = (s + L) ** 2 / (2.0 * ab)
    f_m = (L - M) ** 2 / (2.0 * ab)
    plate = f_m + (s + M) / mx
    fall = 1.0 - (L - s) ** 2 / (2.0 * ab)
    out = np.where(s <= -L, 0.0, np.where(s <= -M, rise, np.where(s <= M, plate, np.where(s <= L, fall, 1.0))))
    return out


@lru_cache(maxsize=4)
def _build(n: int, theta_key: Tuple[float, ...], model: str) -> sp.csr_matrix:
    theta = np.deg2rad(np.asarray(theta_key, dtype=np.float64))
    n_views = theta.size
    n_det = n
    c = (n - 1) / 2.0
    cols_idx, rows_idx = np.meshgrid(np.arange(n), np.arange(n))
    x = (cols_idx - c).ravel()           # x sang phải
    y = (c - rows_idx).ravel()           # y hướng lên
    inside = (x ** 2 + y ** 2) <= (n / 2.0) ** 2   # đĩa nội tiếp (circle=True của radon/iradon)
    pix = np.nonzero(inside)[0]
    xs, ys = x[pix], y[pix]

    rows, cols, vals = [], [], []
    det_c = (n_det - 1) / 2.0
    for a, th in enumerate(theta):
        sc = xs * np.cos(th) + ys * np.sin(th) + det_c      # toạ độ tâm pixel trên trục đầu dò (đơn vị bin)
        if model == "linear":
            i0 = np.floor(sc).astype(np.int64)
            w1 = sc - i0
            cand = ((0, 1.0 - w1), (1, w1))
            for off, w in cand:
                idx = i0 + off
                ok = (idx >= 0) & (idx < n_det) & (w > 0)
                rows.append(a * n_det + idx[ok]); cols.append(pix[ok]); vals.append(w[ok])
        else:  # 'strip'
            ca, sa = abs(np.cos(th)), abs(np.sin(th))
            L = (ca + sa) / 2.0
            M = abs(ca - sa) / 2.0
            ab = max(ca * sa, 1e-12)
            mx = max(ca, sa)
            j0 = np.floor(sc - L + 0.5).astype(np.int64)
            for k in range(3):
                j = j0 + k
                w = _trap_cdf(j + 0.5 - sc, L, M, ab, mx) - _trap_cdf(j - 0.5 - sc, L, M, ab, mx)
                ok = (j >= 0) & (j < n_det) & (w > 1e-7)
                rows.append(a * n_det + j[ok]); cols.append(pix[ok]); vals.append(w[ok])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals).astype(np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_views * n_det, n * n), dtype=np.float32)


def system_matrix(n: int, theta_deg, model: str = "strip") -> sp.csr_matrix:
    """Ma trận hệ thống A (có cache theo (n, theta, model))."""
    return _build(int(n), tuple(float(t) for t in np.asarray(theta_deg).ravel()), model)


class Projector:
    """Bao bọc ma trận A với các thao tác tiện dụng (định dạng sinogram (n_det, n_views))."""

    def __init__(self, n: int, theta_deg, model: str = "strip"):
        self.n = int(n)
        self.theta = np.asarray(theta_deg, dtype=np.float64).ravel()
        self.n_views = self.theta.size
        self.n_det = self.n
        self.model = model
        self.A = system_matrix(self.n, self.theta, model)
        self.AT = self.A.T.tocsr()
        self._blocks = {}

    # --- toàn bộ ---------------------------------------------------------------------
    def forward(self, img: np.ndarray) -> np.ndarray:
        """Ảnh (n, n) -> sinogram (n_det, n_views)."""
        p = self.A @ img.astype(np.float32).ravel()
        return p.reshape(self.n_views, self.n_det).T.astype(np.float64)

    def back(self, sino: np.ndarray) -> np.ndarray:
        """Sinogram (n_det, n_views) -> ảnh chiếu ngược (n, n)."""
        b = self.AT @ sino.T.astype(np.float32).ravel()
        return b.reshape(self.n, self.n).astype(np.float64)

    # --- theo nhóm góc (OS-SART) -----------------------------------------------------
    def block(self, views: np.ndarray):
        """Trả (A_s, A_s^T, rows, rowsum, colsum) cho nhóm góc `views` (có cache)."""
        key = tuple(int(v) for v in views)
        if key not in self._blocks:
            rows = np.concatenate([np.arange(v * self.n_det, (v + 1) * self.n_det) for v in views])
            As = self.A[rows]
            Ast = As.T.tocsr()
            rs = np.asarray(As.sum(axis=1)).ravel()
            cs = np.asarray(As.sum(axis=0)).ravel()
            self._blocks[key] = (As, Ast, rows, rs, cs)
        return self._blocks[key]
