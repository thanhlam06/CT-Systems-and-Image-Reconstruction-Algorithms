"""
pipeline.py - Điều phối: mô phỏng quét -> chạy nhiều thuật toán tái tạo -> tính chỉ số.

Các thuật toán (khoá):
    BP    : chiếu ngược không lọc                         (truyền thống)
    FBP   : chiếu ngược có lọc (ramp / shepp-logan / ...) (truyền thống)
    SART  : ART/SART                                       (lặp - IR)
    SIRT  : SIRT                                           (lặp - IR)
    MBIR  : PWLS + TV (tái tạo lặp theo mô hình thống kê)  (lặp - IR)
    ASIR  : ASIR-like, trộn x% IR + (100-x)% FBP           (lặp - IR lai)
    DLIR  : CNN khử nhiễu trên FBP (mô phỏng DLIR)         (học sâu - DL)
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import recon
from .dl_model import load_model
from .metrics import evaluate
from .physics import ScanData, ScanParams, rho_to_hu, simulate_scan
from .projector import Projector

ALGOS: "OrderedDict[str, tuple]" = OrderedDict([
    ("BP",   ("BP (không lọc)",        "Truyền thống")),
    ("FBP",  ("FBP",                   "Truyền thống")),
    ("SART", ("SART/ART",              "Lặp (IR)")),
    ("SIRT", ("SIRT",                  "Lặp (IR)")),
    ("MBIR", ("MBIR (PWLS-TV)",        "Lặp (IR)")),
    ("ASIR", ("ASIR-like",             "Lặp (IR)")),
    ("DLIR", ("DLIR (CNN)",            "Học sâu (DL)")),
])

DEFAULT_CFG: Dict[str, float] = {
    "fbp_filter": "ramp",
    "sart_iter": 1, "sart_relax": 0.5,
    "sirt_iter": 40,
    "mbir_beta": 150.0, "mbir_outer": 6, "mbir_cg": 12,
    "asir_pct": 50,
    "dl_strength": 1.0,
}


@dataclass
class AlgoResult:
    key: str
    label: str
    group: str
    hu: np.ndarray
    seconds: float
    metrics: Dict[str, float]


def label_for(key: str, cfg: Dict) -> str:
    base = ALGOS[key][0]
    if key == "FBP":
        return f"FBP ({cfg['fbp_filter']})"
    if key == "SART":
        return f"SART ({int(cfg['sart_iter'])} vòng)"
    if key == "SIRT":
        return f"SIRT ({int(cfg['sirt_iter'])} vòng)"
    if key == "MBIR":
        return f"MBIR (β={cfg['mbir_beta']:g})"
    if key == "ASIR":
        return f"ASIR-like {int(cfg['asir_pct'])}%"
    if key == "DLIR":
        return f"DLIR (độ mạnh {cfg['dl_strength']:g})"
    return base


class ReconContext:
    """Giữ các tài nguyên dùng chung (ma trận hệ thống, kết quả trung gian) cho MỘT lần quét."""

    def __init__(self, scan: ScanData):
        self.scan = scan
        self._lin: Optional[Projector] = None
        self._strip: Optional[Projector] = None
        self._memo: Dict[tuple, np.ndarray] = {}

    @property
    def lin(self) -> Projector:
        if self._lin is None:
            self._lin = Projector(self.scan.n, self.scan.theta, "linear")
        return self._lin

    @property
    def strip(self) -> Projector:
        if self._strip is None:
            self._strip = Projector(self.scan.n, self.scan.theta, "strip")
        return self._strip

    def hu(self, key: str, cfg: Dict) -> np.ndarray:
        """Ảnh HU của thuật toán `key` (có nhớ đệm các kết quả trung gian)."""
        s = self.scan
        if key == "BP":
            return rho_to_hu(recon.bp(s.sino, s.theta, self.lin))
        if key == "FBP":
            mk = ("FBP", cfg["fbp_filter"])
            if mk not in self._memo:
                self._memo[mk] = rho_to_hu(recon.fbp(s.sino, s.theta, cfg["fbp_filter"], self.lin))
            return self._memo[mk]
        if key == "SART":
            return rho_to_hu(recon.sart(s.sino, s.theta, int(cfg["sart_iter"]), float(cfg["sart_relax"]), proj=self.strip))
        if key == "SIRT":
            return rho_to_hu(recon.sirt(s.sino, s.theta, int(cfg["sirt_iter"]), proj=self.strip))
        if key == "MBIR":
            mk = ("MBIR", float(cfg["mbir_beta"]), int(cfg["mbir_outer"]), int(cfg["mbir_cg"]))
            if mk not in self._memo:
                self._memo[mk] = rho_to_hu(recon.pwls_tv(s.sino, s.weights, s.theta, float(cfg["mbir_beta"]),
                                                         int(cfg["mbir_outer"]), int(cfg["mbir_cg"]), proj=self.strip))
            return self._memo[mk]
        if key == "ASIR":
            fb = self.hu("FBP", {**cfg, "fbp_filter": "ramp"})
            ir = self.hu("MBIR", cfg)
            return recon.asir_blend(fb, ir, float(cfg["asir_pct"]))
        if key == "DLIR":
            model = load_model()
            if model is None:
                raise RuntimeError("Chưa có trọng số DL (ct_sim/weights/dlir_cnn.npz). Chạy: python train_dl.py")
            fb = self.hu("FBP", {**cfg, "fbp_filter": "ramp"})
            return model.denoise(fb, float(cfg["dl_strength"]))
        raise KeyError(key)

    def run(self, key: str, cfg: Dict) -> AlgoResult:
        t0 = time.perf_counter()
        hu = self.hu(key, cfg)
        dt = time.perf_counter() - t0
        return AlgoResult(key, label_for(key, cfg), ALGOS[key][1], hu, dt,
                          evaluate(hu, self.scan.hu_true, self.scan.info))


def run_algorithms(scan: ScanData, keys: List[str], cfg: Optional[Dict] = None,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   ctx: Optional[ReconContext] = None) -> "OrderedDict[str, AlgoResult]":
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    ctx = ctx or ReconContext(scan)
    out: "OrderedDict[str, AlgoResult]" = OrderedDict()
    # tính FBP/MBIR trước (nền cho ASIR/DLIR) -> thời gian của ASIR/DLIR chỉ tính phần cộng thêm
    for i, k in enumerate(keys):
        if progress:
            progress(i, len(keys), f"Đang tái tạo: {ALGOS[k][0]}")
        if k in ("ASIR", "DLIR"):
            ctx.hu("FBP", {**cfg, "fbp_filter": "ramp"})
            if k == "ASIR":
                ctx.hu("MBIR", cfg)
        out[k] = ctx.run(k, cfg)
    if progress:
        progress(len(keys), len(keys), "Hoàn tất")
    return out


def dose_sweep(name: str, n: int, base: ScanParams, ma_list, keys: List[str], cfg: Optional[Dict] = None,
               metal: int = 0, progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Dict[str, list]]:
    """Quét ALARA: thay đổi mA (liều), tái tạo bằng các thuật toán, ghi lại chỉ số.

    Trả về {key: {"ma": [...], "dose": [...], "PSNR_dB": [...], "SSIM": [...], "Noise_HU": [...], "CNR": [...]}}.
    """
    from dataclasses import replace
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    res: Dict[str, Dict[str, list]] = {k: {"ma": [], "dose": [], "mAs": [], "PSNR_dB": [], "SSIM": [], "Noise_HU": [],
                                           "CNR": [], "RMSE_HU": []} for k in keys}
    for i, ma in enumerate(ma_list):
        if progress:
            progress(i, len(ma_list), f"Quét mA = {ma:g}")
        sc = simulate_scan(name, n, replace(base, ma=float(ma)), metal=metal)
        ctx = ReconContext(sc)
        for k in keys:
            r = ctx.run(k, cfg)
            d = res[k]
            d["ma"].append(float(ma))
            d["dose"].append(sc.dose["CTDIvol_mGy"])
            d["mAs"].append(sc.dose["mAs"])
            for m in ("PSNR_dB", "SSIM", "Noise_HU", "CNR", "RMSE_HU"):
                d[m].append(r.metrics[m])
    if progress:
        progress(len(ma_list), len(ma_list), "Hoàn tất")
    return res
