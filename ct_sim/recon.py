"""
recon.py - Các thuật toán tái tạo ảnh CT.

NHÓM TRUYỀN THỐNG (analytic)
    * bp           : Chiếu ngược không lọc (Backprojection) -> ảnh bị mờ "sương" 1/r
    * fbp          : Chiếu ngược có lọc (Filtered Back Projection) với bộ lọc ramp / shepp-logan / cosine / hamming / hann

NHÓM LẶP (IR - Iterative Reconstruction)
    * sart         : ART/SART (Kaczmarz theo khối góc, OS-SART)
    * sirt         : SIRT (Landweber có tiền điều kiện)
    * pwls_tv      : MBIR - Penalized Weighted Least Squares + regulariser TV (mô hình thống kê Poisson)
    * asir_blend   : ASIR-like - trộn x% FBP + (100-x)% IR (như các hãng thương mại)

Tất cả hàm nhận sinogram `sino` kích thước (n_det, n_views) theo đơn vị pixel-nước và
trả về ảnh `rho` (n, n) với nước = 1.0 (xem physics.py).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .phantoms import circle_mask
from .projector import Projector

# ----------------------------------------------------------------------------------------
# Bộ lọc miền tần số (sao chép đúng định nghĩa của skimage để so khớp với iradon)
# ----------------------------------------------------------------------------------------
FILTERS = ("ramp", "shepp-logan", "cosine", "hamming", "hann")


def fourier_filter(size: int, name: str) -> np.ndarray:
    """Đáp ứng tần số H(w) của bộ lọc. 'ramp' = |w| (Ram-Lak) -> triệt tiêu sương mờ 1/r."""
    n = np.concatenate((np.arange(1, size // 2 + 1, 2, dtype=int), np.arange(size // 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    h = 2 * np.real(np.fft.fft(f))              # |w| rời rạc chính xác (ramp)
    if name == "ramp":
        pass
    elif name == "shepp-logan":
        omega = np.pi * np.fft.fftfreq(size)[1:]
        h[1:] *= np.sin(omega) / omega
    elif name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        h *= np.fft.fftshift(np.sin(freq))
    elif name == "hamming":
        h *= np.fft.fftshift(np.hamming(size))
    elif name == "hann":
        h *= np.fft.fftshift(np.hanning(size))
    else:
        raise ValueError(f"Bộ lọc không hợp lệ: {name}")
    return h[:, None]


def filter_sinogram(sino: np.ndarray, name: str = "ramp") -> np.ndarray:
    """q(s, theta) = F^-1{ H(w) * F{p(s, theta)} }  -> lọc từng hình chiếu 1D (theo trục detector)."""
    n_det = sino.shape[0]
    size = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
    pad = size - n_det
    padded = np.pad(sino, ((0, pad), (0, 0)), mode="constant")
    h = fourier_filter(size, name)
    q = np.real(np.fft.ifft(np.fft.fft(padded, axis=0) * h, axis=0))
    return q[:n_det]


def _dtheta_rad(theta: np.ndarray) -> float:
    th = np.asarray(theta, dtype=np.float64)
    return float(np.deg2rad(th[1] - th[0])) if th.size > 1 else float(np.pi)


# ----------------------------------------------------------------------------------------
# Truyền thống
# ----------------------------------------------------------------------------------------
def fbp(sino: np.ndarray, theta: np.ndarray, filter_name: str = "ramp", proj: Optional[Projector] = None) -> np.ndarray:
    """FBP:  f(x,y) = (dtheta/2) * sum_theta  q_theta( x cos + y sin ).

    Dòng áp dụng bộ lọc Ramp:  `q = filter_sinogram(sino, filter_name)`  (H(w) = |w|).
    """
    n = sino.shape[0]
    proj = proj or Projector(n, theta, "linear")
    q = filter_sinogram(sino, filter_name)           # <-- BỘ LỌC RAMP (triệt tiêu sương mờ 1/r)
    return proj.back(q) * (_dtheta_rad(theta) / 2.0)


def fbp_skimage(sino: np.ndarray, theta: np.ndarray, filter_name: str = "ramp") -> np.ndarray:
    """FBP dùng skimage.transform.iradon (dùng để đối chiếu / kiểm thử)."""
    from skimage.transform import iradon
    rec = iradon(sino, theta=theta, filter_name=filter_name, circle=True)
    span = float(theta[-1] - theta[0] + (theta[1] - theta[0])) if len(theta) > 1 else 180.0
    return rec * (180.0 / span)


def bp(sino: np.ndarray, theta: np.ndarray, proj: Optional[Projector] = None) -> np.ndarray:
    """Chiếu ngược KHÔNG lọc - bảo toàn tổng 'khối lượng' so với FBP (chỉ để minh hoạ độ mờ 1/r)."""
    n = sino.shape[0]
    proj = proj or Projector(n, theta, "linear")
    b = proj.back(sino)
    ref = fbp(sino, theta, "ramp", proj)
    m = circle_mask(n)
    s = ref[m].sum() / max(b[m].sum(), 1e-12)
    return b * s


def fbp_accumulate(sino: np.ndarray, theta: np.ndarray, filter_name: str = "ramp",
                   proj: Optional[Projector] = None, n_frames: Optional[int] = None) -> List[np.ndarray]:
    """Danh sách ảnh FBP CỘNG DỒN theo từng góc chiếu (dùng cho hoạt hình): frame k = tổng k+1 góc đầu."""
    n = sino.shape[0]
    n_views = sino.shape[1]
    proj = proj or Projector(n, theta, "linear")
    q = filter_sinogram(sino, filter_name)
    scale = _dtheta_rad(theta) / 2.0
    acc = np.zeros((n, n))
    frames = []
    for a in range(n_views):
        As, Ast, rows, _, _ = proj.block(np.array([a]))
        acc += (Ast @ q[:, a].astype(np.float32)).reshape(n, n) * scale
        frames.append(acc.copy())
    if n_frames and n_frames < n_views:
        idx = np.unique(np.linspace(0, n_views - 1, n_frames).astype(int))
        return [frames[i] for i in idx]
    return frames


# ----------------------------------------------------------------------------------------
# Lặp: SART / OS-SART
# ----------------------------------------------------------------------------------------
def _stride_order(m: int) -> np.ndarray:
    """Thứ tự duyệt nhóm góc cách đều tối đa (bước ~ 0.618*m, nguyên tố cùng nhau với m)."""
    if m <= 2:
        return np.arange(m)
    target = int(round(0.618 * m))
    for d in range(0, m):
        for st in (target + d, target - d):
            if 1 <= st < m and np.gcd(st, m) == 1:
                return (np.arange(m) * st) % m
    return np.arange(m)


def sart(sino: np.ndarray, theta: np.ndarray, n_iter: int = 3, relax: float = 0.7, n_subsets: Optional[int] = None,
         x0: Optional[np.ndarray] = None, nonneg: bool = True, proj: Optional[Projector] = None,
         callback: Optional[Callable[[int, np.ndarray], None]] = None) -> np.ndarray:
    """(OS-)SART: x <- x + lambda * C_s A_s^T R_s (p_s - A_s x),  duyệt lần lượt từng nhóm góc."""
    n = sino.shape[0]
    proj = proj or Projector(n, theta, "strip")
    n_views = proj.n_views
    m = int(n_subsets or n_views)
    m = max(1, min(m, n_views))
    p_vec = sino.T.astype(np.float32).ravel()
    x = (np.zeros(n * n, np.float32) if x0 is None else x0.astype(np.float32).ravel().copy())
    subsets = [np.arange(s, n_views, m) for s in range(m)]
    order = _stride_order(m)
    for it in range(n_iter):
        for s in order:
            As, Ast, rows, rs, cs = proj.block(subsets[s])
            r = (p_vec[rows] - As @ x) / np.maximum(rs, 1e-3)
            r[rs < 1e-3] = 0.0
            x += relax * (Ast @ r) / np.maximum(cs, 1e-3)
            if nonneg:
                np.maximum(x, 0.0, out=x)
        if callback:
            callback(it, x.reshape(n, n).astype(np.float64))
    return x.reshape(n, n).astype(np.float64)


def sirt(sino: np.ndarray, theta: np.ndarray, n_iter: int = 40, relax: float = 1.0, x0: Optional[np.ndarray] = None,
         proj: Optional[Projector] = None, callback=None) -> np.ndarray:
    """SIRT: x <- x + lambda * C A^T R (p - A x),  R = 1/rowsum, C = 1/colsum (cập nhật đồng thời toàn bộ tia)."""
    n = sino.shape[0]
    proj = proj or Projector(n, theta, "strip")
    p_vec = sino.T.astype(np.float32).ravel()
    ones_x = np.ones(n * n, np.float32)
    rs = np.asarray(proj.A @ ones_x).ravel()
    cs = np.asarray(proj.AT @ np.ones(proj.A.shape[0], np.float32)).ravel()
    R = np.where(rs > 1e-3, 1.0 / np.maximum(rs, 1e-3), 0.0)
    C = np.where(cs > 1e-3, 1.0 / np.maximum(cs, 1e-3), 0.0)
    x = np.zeros(n * n, np.float32) if x0 is None else x0.astype(np.float32).ravel().copy()
    for it in range(n_iter):
        x += relax * C * (proj.AT @ (R * (p_vec - proj.A @ x)))
        np.maximum(x, 0.0, out=x)
        if callback:
            callback(it, x.reshape(n, n).astype(np.float64))
    return x.reshape(n, n).astype(np.float64)


# ----------------------------------------------------------------------------------------
# Lặp thống kê: PWLS + TV (MBIR) bằng "lagged diffusivity" + CG
# ----------------------------------------------------------------------------------------
def _Dx(x):  return x[:, 1:] - x[:, :-1]
def _Dy(x):  return x[1:, :] - x[:-1, :]


def _DxT(v, shape):
    out = np.zeros(shape, v.dtype)
    out[:, :-1] -= v
    out[:, 1:] += v
    return out


def _DyT(v, shape):
    out = np.zeros(shape, v.dtype)
    out[:-1, :] -= v
    out[1:, :] += v
    return out


def pwls_tv(sino: np.ndarray, weights: np.ndarray, theta: np.ndarray, beta: float = 1.0, n_outer: int = 6,
            n_cg: int = 12, delta: float = 0.01, x0: Optional[np.ndarray] = None,
            proj: Optional[Projector] = None, callback=None) -> np.ndarray:
    """MBIR:   x* = argmin  1/2 sum_i w_i (A x - p)_i^2  +  beta * sum_e sqrt((D_e x)^2 + delta^2)

    * w_i = 1/var(p_i)  (trọng số thống kê từ số đếm photon -> tia ít photon bị 'tin' ít hơn)
    * TV làm trơn nhiễu nhưng giữ biên (delta ~ 10 HU = 0.01 theo thang rho)
    * Giải bằng khuếch tán trễ (lagged diffusivity): mỗi vòng ngoài giải hệ tuyến tính
      (A^T W A + beta L(x_k)) x = A^T W p  bằng vài bước CG (khởi động từ nghiệm trước).
    """
    n = sino.shape[0]
    proj = proj or Projector(n, theta, "strip")
    mask = circle_mask(n)
    mflat = mask.ravel().astype(np.float64)
    A, AT = proj.A, proj.AT
    p_vec = sino.T.astype(np.float64).ravel()
    w_vec = weights.T.astype(np.float64).ravel()
    b = mflat * (AT @ (w_vec * p_vec).astype(np.float32)).astype(np.float64)

    if x0 is None:
        x0 = fbp(sino, theta, "hann")
    x = (x0 * mask).ravel().astype(np.float64)

    for outer in range(n_outer):
        xi = x.reshape(n, n)
        cx = 1.0 / np.sqrt(_Dx(xi) ** 2 + delta ** 2)
        cy = 1.0 / np.sqrt(_Dy(xi) ** 2 + delta ** 2)

        def matvec(v, cx=cx, cy=cy):
            vm = v * mflat
            data = AT @ (w_vec * (A @ vm.astype(np.float32))).astype(np.float32)
            vi = vm.reshape(n, n)
            reg = _DxT(cx * _Dx(vi), (n, n)) + _DyT(cy * _Dy(vi), (n, n))
            return mflat * (data.astype(np.float64) + beta * reg.ravel()) + (1.0 - mflat) * v

        op = LinearOperator((n * n, n * n), matvec=matvec, dtype=np.float64)
        x, _ = cg(op, b, x0=x, maxiter=n_cg, rtol=1e-6)
        if callback:
            callback(outer, x.reshape(n, n))
    return np.maximum(x.reshape(n, n), 0.0)


def asir_blend(fbp_img: np.ndarray, ir_img: np.ndarray, pct_ir: float) -> np.ndarray:
    """ASIR-like: ảnh = (1 - a) * FBP + a * IR  với a = pct_ir/100 (ASIR 40% -> a = 0.4)."""
    a = np.clip(pct_ir / 100.0, 0.0, 1.0)
    return (1.0 - a) * fbp_img + a * ir_img

