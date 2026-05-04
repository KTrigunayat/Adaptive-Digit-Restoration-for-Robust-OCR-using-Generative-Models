# Adaptive Digit Restoration for Robust OCR using Generative Models

A hybrid AI pipeline that restores corrupted MNIST handwritten digit images to improve downstream OCR accuracy. It combines classical Digital Image Processing (DIP) with a Variational Autoencoder (VAE) and a Conditional Latent Diffusion Model (LDM) to adaptively denoise images corrupted by Gaussian noise, motion blur, or spatial masking.

---

## Problem Statement

OCR systems perform well on clean data but degrade significantly under real-world distortions like noise, blur, and occlusion. This project addresses that gap by building a restoration pipeline that identifies the corruption type and adaptively reconstructs clean digit representations — improving OCR accuracy on degraded inputs.

---

## Pipeline Overview

```
Corrupted Image (1×28×28)
        │
        ▼
  ┌─────────────┐
  │  DIP Layer  │  Median filter (k=3) + Histogram Equalization
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ VAE Encoder │  → latent z ∈ ℝ⁶⁴  (μ, log σ)
  └─────────────┘
        │         ╲
        ▼           ▼
  ┌───────────┐  ┌──────────────────────┐
  │ Diffusion │  │ Corruption Classifier│  → 1-hot c ∈ ℝ³
  │  Engine   │◄─┤  (gaussian_noise /   │
  │  (DDPM)   │  │   motion_blur /      │
  └───────────┘  │   spatial_masking)   │
        │         └──────────────────────┘
        ▼
  ┌─────────────┐
  │ VAE Decoder │  → Restored Image (1×28×28)
  └─────────────┘
        │
        ▼
  ┌──────────────┐
  │ OCR Classifier│  → Digit label (0–9)
  └──────────────┘
```

---

## Project Structure

```
.
├── config.yaml                          # All hyperparameters and file paths
├── main.py                              # End-to-end pipeline orchestrator
├── requirements.txt
│
├── data/
│   ├── distortion_engine.py             # Synthetic corruption generator
│   ├── raw/                             # Raw MNIST data
│   └── processed/                       # Processed data
│
├── src/
│   ├── preprocessing/
│   │   ├── dip_filters.py               # Median filter + histogram equalization
│   │   └── morphology.py                # Dilation / erosion utilities
│   ├── models/
│   │   ├── vae.py                       # VAE encoder / decoder
│   │   ├── unet.py                      # Conditional U-Net (noise predictor)
│   │   ├── corruption_classifier.py     # Predicts corruption type
│   │   ├── ocr_classifier.py            # CNN for digit recognition (eval only)
│   │   ├── train_vae.py                 # VAE training script
│   │   ├── train_corruption_classifier.py
│   │   └── train_vae.py
│   ├── core/
│   │   ├── diffusion_engine.py          # DDPM forward / reverse process
│   │   └── train_diffusion.py           # Diffusion training script
│   └── utils/
│       ├── config.py                    # Config loader with key validation
│       ├── logger.py
│       ├── metrics.py                   # PSNR, ELBO, OCR accuracy
│       └── visualize.py
│
├── experiments/
│   ├── baseline_ocr_eval.py             # Baseline accuracy measurement
│   └── results/                         # Evaluation JSON output
│
├── tests/
│   ├── unit/
│   │   └── test_dip_filters.py
│   └── integration/
│       ├── test_baseline_eval.py
│       └── test_full_pipeline.py
│
└── checkpoints/                         # Saved model weights (created at training)
```

---

## Components

### Distortion Engine (`data/distortion_engine.py`)

Applies one of three synthetic corruptions to a clean MNIST image:

| Type | Behaviour |
|---|---|
| `gaussian_noise` | Additive Gaussian noise, σ ~ Uniform(0.1, 0.5) |
| `motion_blur` | Convolution with a 3×3 or 5×5 horizontal motion kernel |
| `spatial_masking` | Zeros out a randomly positioned contiguous 8×8 region |

Accepts an optional `seed` parameter for reproducible corruption.

### DIP Layer (`src/preprocessing/dip_filters.py`)

Classical preprocessing applied before VAE encoding:
1. Median filter (kernel size 3) — suppresses salt-and-pepper noise
2. Global histogram equalization — normalizes contrast

### VAE (`src/models/vae.py`)

Encodes 1×28×28 images into a 64-dimensional latent space.

- **Encoder:** 3 stride-2 Conv layers → (μ, log σ) ∈ ℝ⁶⁴
- **Decoder:** Linear → 3 ConvTranspose layers → Sigmoid → 1×28×28
- **Loss:** ELBO = BCE(recon, x, reduction="sum") + β · KL

### Corruption Classifier (`src/models/corruption_classifier.py`)

Lightweight CNN that predicts the corruption type from a preprocessed image and outputs a 1-hot conditioning vector for the diffusion engine.

- Architecture: Conv→ReLU→MaxPool ×2, Flatten, Linear(3136→128)→ReLU→Dropout(0.3), Linear(128→3)
- Target accuracy: ≥ 85% on held-out corrupted images

### Diffusion Engine (`src/core/diffusion_engine.py`)

DDPM-style LDM operating in the VAE latent space, conditioned on corruption type.

- **Forward process:** adds Gaussian noise over T=1000 timesteps (linear schedule, β₁=1e-4, β_T=0.02)
- **Reverse process:** U-Net predicts noise at each step, conditioned on the 1-hot corruption vector
- **Training loss:** MSE between predicted and actual noise; VAE encoder is frozen

### OCR Classifier (`src/models/ocr_classifier.py`)

Pre-trained CNN for digit recognition (0–9). Used only for evaluation — never updated during restoration training.

---

## Setup

```bash
pip install -r requirements.txt
mkdir -p checkpoints experiments/results
```

---

## Training

Run the three training phases in order:

```bash
# Phase 1 — Train VAE on clean MNIST
python src/models/train_vae.py

# Phase 2 — Train Corruption Classifier on synthetically corrupted MNIST
python src/models/train_corruption_classifier.py

# Phase 3 — Train Diffusion Engine (VAE encoder frozen)
python src/core/train_diffusion.py
```

All hyperparameters and checkpoint paths are read from `config.yaml`.

---

## Running the Pipeline

```bash
python main.py
```

This executes the full restoration pipeline and writes an evaluation report to `experiments/results/eval_report.json`.

---

## Evaluation

```bash
python experiments/baseline_ocr_eval.py
```

Measures OCR accuracy on all three corruption types before restoration, establishing the baseline degradation.

### Metrics

| Metric | Description |
|---|---|
| A_clean | OCR accuracy on unmodified MNIST images |
| A_corrupted | OCR accuracy per corruption type (pre-restoration) |
| A_restored | OCR accuracy per corruption type (post-restoration) |
| Mean PSNR | Peak Signal-to-Noise Ratio between restored and clean images (dB) |
| Mean ELBO | VAE Evidence Lower Bound on clean images |

The pipeline is considered successful when A_restored > A_corrupted and mean PSNR(restored) > mean PSNR(corrupted) for all three corruption types.

---

## Configuration

All settings live in `config.yaml`:

```yaml
data:
  raw_dir: data/raw
  distortion: gaussian_noise   # gaussian_noise | motion_blur | spatial_masking
  seed: 42

vae:
  latent_dim: 64
  beta: 1.0
  epochs: 20
  lr: 1.0e-4
  checkpoint: checkpoints/vae.pth

diffusion:
  timesteps: 1000
  beta_start: 1.0e-4
  beta_end: 0.02
  epochs: 30
  checkpoint: checkpoints/diffusion.pth

device: cuda   # falls back to CPU if CUDA unavailable
seed: 42
```

---

## Testing

```bash
python -m pytest tests/ -v
```

29 tests across unit and integration suites covering shape invariants, reproducibility, probability distribution validity, and end-to-end pipeline wiring.

---

## Dataset

[MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/) — 60,000 training / 10,000 test grayscale images of digits 0–9, each 28×28 pixels. Synthetic distortions are applied at runtime; no pre-corrupted dataset is required.

---

## Tech Stack

- **PyTorch** — model training and inference
- **OpenCV** — histogram equalization
- **scikit-image** — median filtering
- **Hypothesis** — property-based testing
- **PyYAML** — configuration management
