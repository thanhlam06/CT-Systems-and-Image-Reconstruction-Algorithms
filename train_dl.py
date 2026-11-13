"""
train_dl.py - Huấn luyện OFFLINE mô hình DLIR đồ chơi (CNN khử nhiễu dạng phần dư) bằng PyTorch.

    python train_dl.py                # huấn luyện đầy đủ (~vài phút trên CPU)
    python train_dl.py --quick        # chạy thử nhanh

Dữ liệu: sinh tự động các phantom ngẫu nhiên -> mô phỏng quét (kV, mA ngẫu nhiên, Poisson, đôi khi beam hardening/kim loại)
-> FBP ramp liều thấp (đầu vào) / ảnh sự thật nền (đầu ra mong muốn).  Phantom huấn luyện KHÔNG trùng 3 phantom của GUI.

Kết quả (trong ct_sim/weights/):  dlir_cnn.npz (trọng số cho NumPy), dlir_train_log.json, dlir_examples.npz
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from ct_sim import recon
from ct_sim.dl_model import HU_CLIP, HU_SCALE, DLIRModel
from ct_sim.metrics import evaluate
from ct_sim.phantoms import HU_METAL, make_phantom, random_phantom, render_replace
from ct_sim.physics import ScanParams, simulate_scan, simulate_scan_from_hu
from ct_sim.phantoms import PhantomInfo

N = 128
C = 24            # số kênh
L = 6             # số lớp tích chập
WDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ct_sim", "weights")

_INFO = PhantomInfo(name="random", fov_cm=25.0, window=(40.0, 400.0),
                    rois={"bg": (0, 0, 0.05), "lesion": (0.1, 0.1, 0.05)})


def make_pair(seed: int):
    """Một cặp huấn luyện: (FBP nhiễu [HU], sự thật nền [HU])."""
    rng = np.random.default_rng(seed)
    hu_hr = random_phantom(2 * N, rng, ss=2)
    if rng.random() < 0.10:  # đôi khi có kim loại
        r, th = rng.uniform(0.0, 0.5), rng.uniform(0, 2 * np.pi)
        m = render_replace(2 * N, [(1.0, 0.05, 0.05, r * np.cos(th), r * np.sin(th), 0)], 0.0, 2)
        hu_hr = hu_hr * (1 - m) + HU_METAL * m
    ma = float(np.exp(rng.uniform(np.log(8.0), np.log(250.0))))
    kv = float(rng.choice([100.0, 120.0, 120.0, 140.0]))
    nv = int(rng.choice([180, 180, 120, 90]))
    bh = bool(rng.random() < 0.2)
    sp = ScanParams(kv=kv, ma=ma, n_views=nv, beam_hardening=bh, seed=int(seed))
    sc = simulate_scan_from_hu(hu_hr, _INFO, sp)
    rec = recon.fbp(sc.sino, sc.theta, "ramp")
    return (1000.0 * (rec - 1.0)).astype(np.float32), sc.hu_true.astype(np.float32)


def build_dataset(n_pairs: int, seed0: int, workers: int):
    t0 = time.time()
    seeds = [seed0 + i for i in range(n_pairs)]
    with mp.Pool(workers) as pool:
        out = pool.map(make_pair, seeds, chunksize=8)
    x = np.stack([o[0] for o in out])
    y = np.stack([o[1] for o in out])
    print(f"  dữ liệu: {n_pairs} cặp trong {time.time() - t0:.0f}s")
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pairs", type=int, default=1400)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.quick:
        args.pairs, args.val, args.steps = 80, 16, 120

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    print("[1/4] Sinh dữ liệu huấn luyện...")
    x_all, y_all = build_dataset(args.pairs + args.val, 10_000, args.workers)
    xt, yt = x_all[: args.pairs], y_all[: args.pairs]
    xv, yv = x_all[args.pairs:], y_all[args.pairs:]

    def norm(a):
        return np.clip(a, *HU_CLIP) / HU_SCALE

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [1] + [C] * (L - 1) + [1]
            self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, padding=0) for i in range(L)])

        def forward(self, x):
            h = x
            for i, cv in enumerate(self.convs):
                h = cv(F.pad(h, (1, 1, 1, 1), mode="replicate"))
                if i < L - 1:
                    h = F.relu(h)
            return x - h

    net = Net()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[2/4] Mô hình: {L} lớp Conv3x3, {C} kênh, {n_par} tham số")
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    Xt = torch.from_numpy(norm(xt))[:, None]
    Yt = torch.from_numpy(norm(yt))[:, None]
    Xv = torch.from_numpy(norm(xv))[:, None]
    Yv = torch.from_numpy(norm(yv))[:, None]
    crop, bs = 64, 32
    rng = np.random.default_rng(1)

    def val_loss():
        net.eval()
        with torch.no_grad():
            out = net(Xv)
            l1 = float(F.l1_loss(out, Yv)) * HU_SCALE
            base = float(F.l1_loss(Xv, Yv)) * HU_SCALE
            # sai số trong vùng mô mềm (|HU| < 150) - phản ánh khử nhiễu hạt
            m = (yv > -150) & (yv < 150)
            mt = torch.from_numpy(m)[:, None]
            soft = float(((out - Yv).abs() * mt).sum() / mt.sum()) * HU_SCALE
            soft_b = float(((Xv - Yv).abs() * mt).sum() / mt.sum()) * HU_SCALE
        net.train()
        return l1, base, soft, soft_b

    print("[3/4] Huấn luyện...")
    log = {"step": [], "train_l1_HU": [], "val_step": [], "val_l1_HU": [], "val_softL1_HU": [],
           "baseline_val_l1_HU": None, "baseline_soft_l1_HU": None}
    t0 = time.time()
    run = 0.0
    val_every = max(1, args.steps // 24)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, Xt.shape[0], bs)
        i0 = rng.integers(0, N - crop + 1, bs)
        j0 = rng.integers(0, N - crop + 1, bs)
        xb = torch.stack([Xt[k, :, a:a + crop, b:b + crop] for k, a, b in zip(idx, i0, j0)])
        yb = torch.stack([Yt[k, :, a:a + crop, b:b + crop] for k, a, b in zip(idx, i0, j0)])
        if rng.random() < 0.5:
            xb, yb = xb.flip(-1), yb.flip(-1)
        if rng.random() < 0.5:
            xb, yb = xb.flip(-2), yb.flip(-2)
        loss = F.l1_loss(net(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % 20 == 0:
            log["step"].append(step)
            log["train_l1_HU"].append(run * HU_SCALE)
        if step % val_every == 0 or step == args.steps:
            v, base, soft, soft_b = val_loss()
            log["val_step"].append(step)
            log["val_l1_HU"].append(v)
            log["val_softL1_HU"].append(soft)
            log["baseline_val_l1_HU"], log["baseline_soft_l1_HU"] = base, soft_b
            print(f"  step {step:5d}/{args.steps}  train L1 {run * HU_SCALE:6.2f} HU | val L1 {v:6.2f} (FBP {base:6.2f}) "
                  f"| mô mềm {soft:5.2f} (FBP {soft_b:5.2f})  [{time.time() - t0:.0f}s]")

    # ----- lưu trọng số (NumPy) -----
    os.makedirs(WDIR, exist_ok=True)
    params = {}
    for i, cv in enumerate(net.convs):
        params[f"w{i}"] = cv.weight.detach().numpy().astype(np.float32)
        params[f"b{i}"] = cv.bias.detach().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(WDIR, "dlir_cnn.npz"), **params)
    log.update({"n_params": n_par, "layers": L, "channels": C, "pairs": args.pairs, "val_pairs": args.val,
                "steps": args.steps, "batch": bs, "crop": crop, "seconds": time.time() - t0,
                "torch": torch.__version__, "image_size": N})
    with open(os.path.join(WDIR, "dlir_train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)

    # ----- ví dụ minh hoạ (3 phantom của GUI, liều thấp) -----
    print("[4/4] Tạo ví dụ minh hoạ...")
    model = DLIRModel(params)
    ex = {}
    for key, name, ma in (("head", "Đầu (HU)", 25.0), ("sl", "Shepp-Logan", 25.0), ("chest", "Lồng ngực (HU)", 40.0)):
        sc = simulate_scan(name, N, ScanParams(ma=ma, seed=7))
        fb = 1000.0 * (recon.fbp(sc.sino, sc.theta, "ramp") - 1.0)
        ex[key + "_in"] = fb.astype(np.float32)
        ex[key + "_out"] = model.denoise(fb).astype(np.float32)
        ex[key + "_gt"] = sc.hu_true.astype(np.float32)
        m_in, m_out = evaluate(fb, sc.hu_true, sc.info), evaluate(ex[key + "_out"], sc.hu_true, sc.info)
        print(f"  {name:16s} @{ma:.0f} mA: noise {m_in['Noise_HU']:.1f} -> {m_out['Noise_HU']:.1f} HU,"
              f" SSIM {m_in['SSIM']:.3f} -> {m_out['SSIM']:.3f}, PSNR {m_in['PSNR_dB']:.2f} -> {m_out['PSNR_dB']:.2f}")
    np.savez_compressed(os.path.join(WDIR, "dlir_examples.npz"), **ex)
    print("Xong. Trọng số:", os.path.join(WDIR, "dlir_cnn.npz"))


if __name__ == "__main__":
    mp.freeze_support()
    main()
