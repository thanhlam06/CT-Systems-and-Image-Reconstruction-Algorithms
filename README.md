# CT Systems and Image Reconstruction Algorithms (`CT_Simulator`)

A modular Medical Computed Tomography (CT) Simulation & Clinical Image Analysis Workstation built in Python. This project models the complete physical pipeline of an X-ray CT scanner—from polychromatic Beer–Lambert attenuation, quantum Poisson / electronic readout noise, and ALARA radiation dose metrics ($\text{CTDI}_{\text{vol}}$, $\text{DLP}$) to analytical, algebraic, statistical iterative, and deep learning reconstruction algorithms.

---

## Key Features

### 1. X-Ray Physics & Dose Modeling (`ct_sim/physics.py`, `ct_sim/phantoms.py`)
- **Polychromatic Kramer/Tungsten Spectrum & Beer–Lambert Attenuation**: Simulates effective linear attenuation $\mu(x,y,E)$, beam hardening, and metal implant photon starvation.
- **Quantum Statistics & Readout Noise**: Models discrete photon counts $N_i \sim \text{Poisson}(\bar{N}_i) + \mathcal{N}(0, \sigma_e^2)$ as a function of tube voltage ($\text{kVp}$) and tube current ($\text{mA}$).
- **Dosimetry & ALARA**: Computes $\text{CTDI}_{\text{vol}}$ ($\text{mGy}$) and $\text{DLP}$ ($\text{mGy}\cdot\text{cm}$) with quantitative verification of Rose's criterion ($\text{SNR} \ge 5$).
- **Anatomical Phantoms**: Calibrated Hounsfield Unit ($\text{HU}$) phantoms for Head/Brain, Thorax/Lungs, Abdomen/Liver, and Dental/Metal implants.

### 2. Forward Projector & 7 Reconstruction Algorithms (`ct_sim/projector.py`, `ct_sim/recon.py`, `ct_sim/dl_model.py`)
- **Anti-Inverse-Crime Architecture**: Decouples data generation (high-resolution Fourier/upsampled Radon transform) from iterative reconstruction (analytical Strip-Integral Trapezoidal CDF footprint matrix $A_{ij}$).
- **7 Reconstruction Methods**:
  1. **BP (Unfiltered Back-Projection)**: Demonstrates $1/(\pi r)$ PSF blurring.
  2. **FBP (Filtered Back-Projection)**: Supports 5 frequency-domain apodization windows (`ramp` / Ram-Lak, `shepp-logan`, `cosine`, `hamming`, `hann`).
  3. **SART (Ordered-Subsets Simultaneous Algebraic Reconstruction Technique)**: Golden-ratio angular subset ordering ($\Delta\phi \approx 111.246^\circ$) with non-negativity projection $\mathcal{P}_{\ge 0}$.
  4. **SIRT (Simultaneous Iterative Reconstruction Technique)**: Diagonal preconditioned Landweber iteration.
  5. **MBIR (`PWLS-TV`)**: Penalized Weighted Least-Squares with edge-preserving Total Variation regularization solved via Lagged Diffusivity Fixed-Point + Preconditioned Conjugate Gradient (PCG).
  6. **ASIR / AIDR 3D**: Statistical hybrid blending between FBP and MBIR optimizing Noise Power Spectrum (NPS) texture.
  7. **DLIR (Deep Learning Image Reconstruction)**: Pre-trained 4-layer Residual Convolutional Neural Network (`ct_sim/weights/dlir_cnn.npz`) with edge-guided noise residual estimation.

### 3. Real Clinical CT Processing Engine (`ct_sim/real_ct.py`, `ct_sim/metrics.py`)
- **Piecewise Linear HU Calibration**: Converts clinical DICOM/8-bit/16-bit CT slices into physical Hounsfield Unit maps $\text{HU}(x,y)$ and linear attenuation maps $\mu(x,y)$.
- **Automatic Morphological ROI Detection**: Automatically locates background parenchyma and lesion ROIs to compute quantitative metrics ($\text{RMSE}$, $\text{PSNR}$, $\text{SSIM}$, $\text{Noise SD}$, $\text{Bias}$, $\text{CNR}$).

### 4. Medical Pro Workstation GUI (`gui.py`)
- **HiDPI PyQt6 & `qdarktheme` Interface**: Dark-slate medical workstation with interactive Window/Level ($\text{WL}/\text{WW}$) presets, real-time gantry scan animation, interactive left/right-click ROI placement, multi-algorithm benchmark comparison, and ALARA dose-sweep curves.

---

## Project Structure

```text
CT_Simulator/
├── assets/
│   ├── bk_logo.png
│   └── real_ct/                 # Clinical CT sample cases (Liver, Brain, Chest HRCT)
├── ct_sim/
│   ├── __init__.py
│   ├── physics.py               # Beer-Lambert, Poisson/Gaussian noise, CTDIvol/DLP
│   ├── phantoms.py              # Analytical HU anatomical phantoms
│   ├── projector.py             # Strip-Integral Trapezoidal CDF forward/backprojector
│   ├── recon.py                 # BP, FBP (5 filters), OS-SART, SIRT, MBIR, ASIR
│   ├── dl_model.py              # Residual CNN architecture & inference for DLIR
│   ├── real_ct.py               # Clinical CT HU calibration & automatic ROI detection
│   ├── metrics.py               # RMSE, PSNR, SSIM, ROI Noise SD, Bias, CNR
│   ├── pipeline.py              # End-to-end simulation & ALARA dose-sweep controller
│   └── weights/
│       └── dlir_cnn.npz         # Pre-trained Residual CNN weights
├── report/
│   ├── generate_flowcharts.py   # High-DPI STIX LaTeX algorithm flowchart generator
│   ├── generate_btl_pdf.py      # Academic technical report PDF compiler
│   ├── figures/                 # Generated flowcharts & experimental figures
│   └── BaoCao_BTL_Project2_CT_Scanner_v3_Final.pdf
├── gui.py                       # PyQt6 Medical Pro Workstation GUI
├── main.py                      # Application entry point
├── run_gui.bat                  # One-click Windows launcher
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```
