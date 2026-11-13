"""
physics.py - Mô hình vật lý thu nhận CT: ống tia X (kV, mA), Beer-Lambert, nhiễu lượng tử Poisson,
beam hardening (tuỳ chọn), ước tính liều (ALARA).

Quy ước đơn vị
--------------
* Ảnh tái tạo được biểu diễn theo "mật độ tương đối so với nước"  rho = mu / mu_water = 1 + HU/1000.
  (nước = 1, không khí = 0). Sinogram `p` tương ứng là tích phân đường của rho theo đơn vị "pixel-nước".
* Đại lượng vật lý thật:   p_phys = mu_water(E_eff) * pixel_cm * p       (không thứ nguyên, vào hàm mũ Beer-Lambert)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from skimage.transform import radon

from .phantoms import PhantomInfo, make_phantom

# ----------------------------------------------------------------------------------------
# Hệ số suy giảm khối của nước, nhôm (NIST XCOM, cm^2/g) -> nội suy log-log
# ----------------------------------------------------------------------------------------
_E_TAB = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0])           # keV
_MU_W_TAB = np.array([0.8096, 0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707, 0.1505])  # nước (rho=1 g/cm3) -> cm^-1
_MU_AL_TAB = np.array([3.441, 1.128, 0.5685, 0.3681, 0.2778, 0.2018, 0.1704, 0.1378])   # nhôm (cm^2/g)
RHO_AL = 2.699

PHOTONS_PER_MAS_120 = 2.0e6   # số photon / bin đầu dò / mAs tại 120 kVp (hằng số hiệu chuẩn mô hình)
BH_EXPONENT = 2.8             # phần "dư" (xương/kim loại) suy giảm ~ E^-2.8 (hiệu ứng quang điện chiếm ưu thế)
FILTER_AL_MM = 8.0            # lọc nội tại + bow-tie tương đương nhôm (mm)


def _loglog(x, xs, ys):
    return np.exp(np.interp(np.log(x), np.log(xs), np.log(ys)))


def mu_water(e_kev) -> np.ndarray:
    """Hệ số suy giảm tuyến tính của nước (cm^-1) tại năng lượng E (keV)."""
    return _loglog(np.asarray(e_kev, dtype=np.float64), _E_TAB, _MU_W_TAB)


def mu_aluminium(e_kev) -> np.ndarray:
    return _loglog(np.asarray(e_kev, dtype=np.float64), _E_TAB, _MU_AL_TAB) * RHO_AL


def spectrum(kv: float, n_bins: int = 10):
    """Phổ tia X gần đúng (Kramers) sau lọc nhôm, trọng số đầu dò tích phân năng lượng.

    Phi(E) ~ (kV - E)/E * exp(-mu_Al(E) * t_Al);   w(E) ~ Phi(E) * E   (đầu dò EID)
    -> Điện áp kV quyết định E_max, và do đó phổ năng lượng + năng lượng hiệu dụng.
    """
    e = np.linspace(22.0, kv - 2.0, n_bins)
    phi = (kv - e) / e * np.exp(-mu_aluminium(e) * FILTER_AL_MM / 10.0)
    w = phi * e
    return e, w / w.sum()


def effective_energy(kv: float, path_cm: float = 20.0) -> float:
    """Năng lượng hiệu dụng (keV): E sao cho mu_water(E) bằng suy giảm hiệu dụng của chùm sau `path_cm` nước."""
    e, w = spectrum(kv)
    mu_eff = -np.log(np.sum(w * np.exp(-mu_water(e) * path_cm))) / path_cm
    grid = np.linspace(25.0, kv, 600)
    return float(grid[np.argmin(np.abs(mu_water(grid) - mu_eff))])


# ----------------------------------------------------------------------------------------
# Tham số quét
# ----------------------------------------------------------------------------------------
@dataclass
class ScanParams:
    kv: float = 120.0               # [kV]  điện áp đỉnh bóng phát (kVp): quyết định phổ & mu(E)
    ma: float = 200.0               # [mA]  dòng ống: quyết định số photon I0
    rot_time: float = 1.0           # [s]   thời gian một vòng quay (tích mA * s = mAs)
    n_views: int = 180              # số góc chiếu trong khoảng angular_range
    angular_range: float = 180.0    # [độ]  tầm góc quét (180 = đầy đủ cho chùm song song)
    poisson: bool = True            # nhiễu lượng tử (Poisson)
    electronic_sigma: float = 2.0   # nhiễu điện tử của DAS (rms, đơn vị photon)
    beam_hardening: bool = False    # mô phỏng đa năng (polychromatic)
    seed: Optional[int] = 0

    @property
    def mas_total(self) -> float:
        """Tổng mAs thực tế (tube-on): mA * t_view * n_views, với t_view = rot_time / 180."""
        return self.ma * self.rot_time / 180.0 * self.n_views

    @property
    def mas_per_view(self) -> float:
        return self.ma * self.rot_time / 180.0

    @property
    def i0(self) -> float:
        """Số photon tới I0 trên mỗi bin đầu dò mỗi góc chiếu:  I0 ~ mAs/view * (kV/120)^2."""
        return PHOTONS_PER_MAS_120 * self.mas_per_view * (self.kv / 120.0) ** 2

    def theta(self) -> np.ndarray:
        return np.linspace(0.0, self.angular_range, self.n_views, endpoint=False)


# ----------------------------------------------------------------------------------------
# Kết quả quét
# ----------------------------------------------------------------------------------------
@dataclass
class ScanData:
    params: ScanParams
    info: PhantomInfo
    n: int
    theta: np.ndarray
    hu_true: np.ndarray        # (n, n) sự thật nền (HU)
    sino: np.ndarray           # (n_det, n_views) sinogram ĐO (có nhiễu), đơn vị pixel-nước
    sino_clean: np.ndarray     # sinogram lý tưởng (không nhiễu)
    counts: np.ndarray         # số đếm photon đo được
    weights: np.ndarray        # trọng số thống kê w = 1/var(p) (cho PWLS)
    i0: float
    px_cm: float
    unit: float                # mu_water(E_eff) * px_cm
    e_eff: float
    dose: Dict[str, float] = field(default_factory=dict)

    @property
    def rho_true(self) -> np.ndarray:
        return 1.0 + self.hu_true / 1000.0


def estimate_dose(params: ScanParams, info: PhantomInfo) -> Dict[str, float]:
    """Ước tính liều (mô hình đơn giản, chỉ mang tính tham khảo giáo dục).

    CTDIvol ~ k * (kV/120)^2 * mAs ;  DLP = CTDIvol * L ;  E = k_eff * DLP.
    """
    ctdi = info.ctdi_per_mas_120 * (params.kv / 120.0) ** 2 * params.mas_total
    dlp = ctdi * info.scan_length_cm
    eff = dlp * info.k_eff_dose
    ref = info.ctdi_per_mas_120 * 200.0
    return {
        "mAs": params.mas_total,
        "I0": params.i0,
        "CTDIvol_mGy": ctdi,
        "DLP_mGycm": dlp,
        "E_mSv": eff,
        "rel_dose_pct": 100.0 * ctdi / ref,
    }


# ----------------------------------------------------------------------------------------
def _radon_binned(rho_hr: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Radon trên ảnh độ phân giải 2N, gộp từng cặp bin đầu dò -> (N, n_views), đơn vị pixel-N."""
    s = radon(rho_hr, theta=theta, circle=True)
    return 0.5 * (s[0::2] + s[1::2]) / 2.0


def simulate_scan(name: str, n: int, params: ScanParams, metal: int = 0) -> ScanData:
    """Mô phỏng chuỗi thu nhận CT cho phantom có tên `name` (xem simulate_scan_from_hu)."""
    hu_hr, info = make_phantom(name, 2 * n, metal=metal)
    return simulate_scan_from_hu(hu_hr, info, params)


def simulate_scan_from_hu(hu_hr: np.ndarray, info: PhantomInfo, params: ScanParams) -> ScanData:
    """Mô phỏng toàn bộ chuỗi thu nhận CT từ ảnh HU độ phân giải 2N.

    1. Phantom (HU) ở độ phân giải 2N (tránh inverse-crime) và N (sự thật nền = trung bình khối).
    2. Chiếu xuôi (Radon) -> tích phân đường mu*dl.
    3. Beer-Lambert:  N = I0 * exp(-p)          (đơn năng)   hoặc   I0 * sum_E w_E exp(-p_E)  (đa năng)
    4. Nhiễu Poisson + nhiễu điện tử, sàn 1 photon (photon starvation).
    5. Log:  p_meas = -ln(N / I0)               -> sinogram.
    """
    rng = np.random.default_rng(params.seed)
    n = hu_hr.shape[0] // 2
    hu_true = hu_hr.reshape(n, 2, n, 2).mean(axis=(1, 3))  # khối trung bình -> N x N (sự thật nền)
    theta = params.theta()

    px_cm = info.fov_cm / n
    e_eff = effective_energy(params.kv)
    mu_w = float(mu_water(e_eff))
    unit = mu_w * px_cm   # [1/pixel-nước]

    rho_hr = np.clip(1.0 + hu_hr / 1000.0, 0.0, None)
    if params.beam_hardening:
        rho_w = np.clip(rho_hr, 0.0, 1.0)                # phần "tương đương nước"
        rho_x = np.clip(rho_hr - 1.0, 0.0, None)         # phần dư: xương / kim loại (Z cao)
        Pw = _radon_binned(rho_w, theta)
        Px = _radon_binned(rho_x, theta)
        e, w = spectrum(params.kv)
        trans = np.zeros_like(Pw)
        for ek, wk in zip(e, w):
            p_e = float(mu_water(ek)) * px_cm * Pw + unit * (e_eff / ek) ** BH_EXPONENT * Px
            trans += wk * np.exp(-p_e)
        # hiệu chuẩn nước (water calibration): chùm đa năng qua 20cm nước -> mu hiệu dụng = mu_w(E_eff)
    else:
        P = _radon_binned(rho_hr, theta)
        trans = np.exp(-unit * P)                          # Beer-Lambert: I/I0 = exp(-mu*L)

    i0 = params.i0
    lam = i0 * trans                                       # số photon kỳ vọng tại đầu dò
    if params.poisson:
        counts = rng.poisson(lam).astype(np.float64)       # nhiễu lượng tử (Poisson)
    else:
        counts = lam.copy()
    if params.electronic_sigma > 0 and params.poisson:
        counts = counts + rng.normal(0.0, params.electronic_sigma, counts.shape)
    counts = np.maximum(counts, 1.0)                       # sàn 1 photon: tránh log(0) -> vệt sọc kim loại

    sino = -np.log(counts / i0) / unit                      # log-converter tại detector: p = -ln(I/I0)
    sino_clean = -np.log(np.clip(trans, 1e-12, None)) / unit
    weights = (counts ** 2 / (counts + params.electronic_sigma ** 2)) * unit ** 2

    return ScanData(
        params=params, info=info, n=n, theta=theta, hu_true=hu_true, sino=sino, sino_clean=sino_clean,
        counts=counts, weights=weights, i0=i0, px_cm=px_cm, unit=unit, e_eff=e_eff,
        dose=estimate_dose(params, info),
    )


def rho_to_hu(rho: np.ndarray) -> np.ndarray:
    return 1000.0 * (np.asarray(rho) - 1.0)
