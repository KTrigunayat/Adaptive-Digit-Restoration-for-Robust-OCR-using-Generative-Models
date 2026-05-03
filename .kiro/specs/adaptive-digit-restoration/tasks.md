# Implementation Plan: Adaptive Digit Restoration

## Overview

Incremental implementation across 7 phases: update the distortion engine and DIP layer, train the VAE and corruption classifier, extend the UNet with conditioning and train the diffusion engine, wire the full pipeline in `main.py`, add evaluation utilities, and cover everything with unit and property-based tests using Hypothesis.

## Tasks

- [ ] 1. Update DistortionEngine to use canonical corruption type names
  - In `data/distortion_engine.py`, rename the dispatch keys from `"noise"`, `"blur"`, `"mask"` to `"gaussian_noise"`, `"motion_blur"`, `"spatial_masking"`
  - Update `add_gaussian_noise` to draw σ ~ Uniform(0.1, 0.5) per call (replacing the fixed `var` kwarg default)
  - Update `add_blur` to use a horizontal motion kernel (3×3 or 5×5) instead of Gaussian blur, matching the motion_blur spec
  - Update `add_masking` to zero out exactly one contiguous 8×8 region at a random position instead of random pixel masking
  - Add `seed: int | None = None` parameter to `apply_distortion`; set `np.random.seed(seed)` before sampling when provided
  - Update `config.yaml`: change `data.distortion` from `"noise"` to `"gaussian_noise"` and add the full schema from the design (vae.beta, vae.checkpoint, corruption_classifier section, evaluation section, seed key)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 9.1, 9.2_

- [ ]* 1.1 Write property tests for DistortionEngine
  - **Property 1: Distortion preserves image shape** — for any (28, 28) uint8 array and any valid corruption type, `apply_distortion` output shape equals input shape
  - **Validates: Requirements 1.4**
  - **Property 2: Distortion reproducibility with seed** — same seed + same image + same type produces byte-identical arrays
  - **Validates: Requirements 1.6, 9.4**
  - **Property 3: Spatial masking zeros exactly one 8×8 contiguous block** — output has exactly 64 zeros forming a contiguous 8×8 rectangle; all other pixels unchanged
  - **Validates: Requirements 1.3**
  - Add edge-case unit test: `apply_distortion` with an unknown type raises `ValueError`
  - _File: `tests/unit/test_distortion_engine.py`_

- [ ] 2. Add `preprocess` function to DIP Layer
  - In `src/preprocessing/dip_filters.py`, add a `preprocess(image: np.ndarray) -> np.ndarray` function that chains `median_filter(image, kernel_size=3)` → `cv2.equalizeHist`
  - Ensure the function accepts uint8 or float32 input (convert float32 → uint8 before `equalizeHist`, convert back if needed)
  - Keep the existing `apply_filter` dispatcher unchanged
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 2.1 Write property tests for DIP Layer
  - **Property 4: DIP preprocessing preserves spatial dimensions** — for any (H, W) array with valid pixel values, `preprocess` output shape equals input shape
  - **Validates: Requirements 2.4**
  - Add unit test verifying median filter is applied before histogram equalization (mock or inspect intermediate state)
  - _File: `tests/unit/test_dip_filters.py`_

- [ ] 3. Verify and complete VAE implementation
  - Confirm `src/models/vae.py` matches the design spec: encoder produces `(mu, logvar)` each of shape `(B, 64)`, decoder produces `(B, 1, 28, 28)` with Sigmoid output
  - Add `encode` and `decode` as explicit public methods (currently only `forward` and `reparameterize` exist)
  - Verify `VAE.loss` uses `binary_cross_entropy` with `reduction="sum"` plus β·KL; add `beta` parameter (default 1.0)
  - Add a training script `src/models/train_vae.py` that trains the VAE on clean MNIST using Adam lr=1e-4 for the configured number of epochs and saves weights to `checkpoints/vae.pth`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ]* 3.1 Write property tests for VAE
  - **Property 5: VAE encode-decode shape invariant** — for any batch of (B, 1, 28, 28) float32 tensors, encoder returns `(mu, logvar)` each `(B, 64)`, decoder returns `(B, 1, 28, 28)` with all values in [0, 1]
  - **Validates: Requirements 3.1, 3.4**
  - Add unit tests: architecture has exactly 3 stride-2 conv layers in encoder; `loss` formula matches ELBO definition; `encoder.requires_grad_(False)` disables gradients
  - _File: `tests/unit/test_vae.py`_

- [ ] 4. Implement CorruptionClassifier
  - Create `src/models/corruption_classifier.py` with `CorruptionClassifier(nn.Module)` matching the design architecture: Conv→ReLU→MaxPool × 2, Flatten, Linear(3136→128)→ReLU→Dropout(0.3), Linear(128→3)
  - `forward` returns `(B, 3)` softmax probabilities
  - `predict_onehot` returns `(B, 3)` 1-hot tensor (argmax → scatter)
  - Add a training script `src/models/train_corruption_classifier.py` that generates synthetically corrupted MNIST images using `apply_distortion` for all three types, trains with cross-entropy loss using Adam lr=1e-4, and saves weights to `checkpoints/corruption_classifier.pth`
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 4.1 Write property tests for CorruptionClassifier
  - **Property 6: Classifier output is a valid probability distribution** — for any batch of (B, 1, 28, 28) tensors (including random noise, all-zeros, all-ones), `forward` returns `(B, 3)` where each row sums to 1.0 ± 1e-5 and all values ∈ [0, 1]; `predict_onehot` returns `(B, 3)` with exactly one 1 per row
  - **Validates: Requirements 4.1, 4.2, 4.4**
  - Add smoke test: classifier achieves > random baseline (> 33%) on a small held-out corrupted set after training
  - _File: `tests/unit/test_corruption_classifier.py`_

- [ ] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Extend UNet with corruption conditioning
  - In `src/models/unet.py`, add `ConditioningEmbedding(nn.Module)` with `proj = nn.Sequential(nn.Linear(cond_dim, t_dim), nn.SiLU())` where `cond_dim=3`, `t_dim=64`
  - Update `UNet.__init__` to accept optional `cond_dim: int = 3` and instantiate `self.cond_emb = ConditioningEmbedding(cond_dim, t_dim)`
  - Update `UNet.forward(self, x, t, c=None)`: compute `cond_emb = self.cond_emb(c)` when `c` is provided and add it to `t_emb` before passing to each `ConvBlock`
  - Keep backward compatibility: when `c=None`, skip conditioning (pure timestep embedding)
  - _Requirements: 5.2, 5.3_

- [ ] 7. Update DiffusionEngine with conditioning and VAE integration
  - In `src/core/diffusion_engine.py`, update `__init__` to accept a `vae` parameter (optional, used during training)
  - Update `forward_process` to handle 1D latent vectors `(B, 64)` — change `view(-1, 1, 1, 1)` to `view(-1, 1)` for latent-space operation
  - Update `reverse_process(self, z_t, c)` to pass conditioning vector `c` to `self.unet(x, t_tensor, c)`
  - Update `compute_loss(self, z0, c)` to encode `z0` via frozen VAE encoder when `vae` is provided, then run forward process and pass `c` to UNet
  - Add a training script `src/core/train_diffusion.py` that loads frozen VAE, trains the diffusion engine on corrupted MNIST latents conditioned on corruption type, and saves UNet weights to `checkpoints/diffusion.pth`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [ ]* 7.1 Write property tests for DiffusionEngine
  - **Property 7: DDPM forward process produces correctly scaled noisy latents** — for any `z0 ∈ ℝ⁶⁴` and `t ∈ [0, T)`, verify `E[z_t] ≈ √ᾱ_t · z0` and `Var[z_t] ≈ (1 − ᾱ_t)` over many samples
  - **Validates: Requirements 5.1**
  - **Property 8: Diffusion reverse process output shape** — for any `(B, 64)` noisy latent and `(B, 3)` 1-hot conditioning, `reverse_process` returns shape `(B, 64)`
  - **Validates: Requirements 5.6**
  - **Property 9: VAE encoder remains frozen during diffusion training** — after `compute_loss` + `loss.backward()`, VAE encoder parameter values are identical to pre-backward values
  - **Validates: Requirements 5.5**
  - Add unit tests: configurable T; MSE loss formula; noise schedule β values at t=0 and t=T-1
  - _File: `tests/unit/test_diffusion_engine.py`_

- [ ] 8. Wire end-to-end pipeline in main.py
  - Rewrite `run_pipeline(cfg: dict) -> dict` in `main.py` to execute stages in order: DIP `preprocess` → VAE `encode` → `CorruptionClassifier.predict_onehot` → `DiffusionEngine.reverse_process(z, c)` → VAE `decode`
  - Load all four model checkpoints from paths in `cfg`; raise `FileNotFoundError` if any checkpoint is missing
  - Add CUDA fallback warning when `cfg["device"] == "cuda"` but CUDA is unavailable
  - Log each stage name and output tensor shape at INFO level
  - Return `{"a_clean": ..., "a_corrupted": ..., "a_restored": ..., "mean_psnr": ..., "mean_elbo": ...}` dict
  - Validate that all stage outputs maintain input batch size `B`; raise `RuntimeError` with stage name and shapes on mismatch
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ]* 8.1 Write property tests for the pipeline
  - **Property 10: Pipeline output shape matches input batch size** — for any batch of B corrupted (B, 1, 28, 28) images, pipeline output shape is `(B, 1, 28, 28)` with pixel values in [0, 1]
  - **Validates: Requirements 6.2**
  - **Property 13: Seed reproducibility across full pipeline** — two runs with the same seed and same input produce byte-identical output tensors
  - **Validates: Requirements 9.4**
  - Add edge-case unit test: `FileNotFoundError` raised when a checkpoint path does not exist
  - _File: `tests/unit/test_pipeline.py`_

- [ ] 9. Implement metrics utilities
  - Create `src/utils/metrics.py` with three pure functions:
    - `compute_psnr(restored: Tensor, clean: Tensor) -> float` — returns dB value; returns `float("inf")` for identical inputs
    - `compute_elbo(vae: VAE, images: Tensor) -> float` — runs VAE forward pass and returns mean ELBO
    - `compute_ocr_accuracy(ocr: OCRClassifier, images: Tensor, labels: Tensor) -> float` — returns fraction correct
  - Define `EvalReport` dataclass in `src/utils/metrics.py` with fields `a_clean`, `a_corrupted`, `a_restored`, `mean_psnr`, `mean_elbo`
  - _Requirements: 7.2, 7.3, 7.4, 7.5_

- [ ]* 9.1 Write property tests for metrics
  - **Property 11: PSNR is finite and positive for non-identical images** — for any pair of same-shape tensors with different values, `compute_psnr` returns a finite positive float; for identical tensors, returns `float("inf")`
  - **Validates: Requirements 7.3**
  - **Property 12: Evaluation report contains all required fields** — for any evaluation run, returned `EvalReport` has all five fields with finite numeric values
  - **Validates: Requirements 7.5**
  - _File: `tests/unit/test_metrics.py`_

- [ ] 10. Extend baseline evaluation script
  - Update `experiments/baseline_ocr_eval.py` to evaluate OCR on all three corruption types independently using `apply_distortion`
  - Record `a_clean` and `a_corrupted` per corruption type as a dict
  - Write results to JSON at the path from `cfg["evaluation"]["output_path"]`; create parent directories if needed
  - Import and use `compute_ocr_accuracy` from `src/utils/metrics.py`
  - _Requirements: 7.6, 8.1, 8.2, 8.3, 8.4_

- [ ] 11. Update configuration and reproducibility utilities
  - Update `config.yaml` to match the full schema from the design: add `vae.beta`, `vae.checkpoint`, `corruption_classifier` section, `evaluation.output_path`, top-level `seed`; update `data.distortion` to `"gaussian_noise"`; update `preprocessing.filter` to `"median"` with `kernel_size: 3`
  - Update `src/utils/config.py` `load_config` to validate required keys (`data.raw_dir`, `vae.latent_dim`, `diffusion.timesteps`, `diffusion.beta_start`, `diffusion.beta_end`) and raise `KeyError` with the missing key name
  - Apply seed from config to `torch`, `numpy`, and `random` at startup in `main.py`
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ]* 11.1 Write unit tests for config loading
  - Test that `load_config` raises `KeyError` when each required key is missing
  - Test that seed is applied to all three RNG sources
  - _File: `tests/unit/test_config.py`_

- [ ] 12. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Set up tests directory and integration tests
  - Create `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
  - Create `tests/integration/test_baseline_eval.py`: run `baseline_ocr_eval.main()` with a small synthetic dataset (no real MNIST download) and assert the output JSON has the correct structure
  - Create `tests/integration/test_full_pipeline.py`: end-to-end smoke test with randomly initialized models (no trained weights) verifying stage ordering via mocks and that output shape is `(B, 1, 28, 28)`
  - _Requirements: 6.1, 8.4_

- [ ] 14. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `hypothesis` with `hypothesis[numpy]`; each test is tagged with a comment referencing the design property number (e.g., `# Property 1: Distortion preserves image shape`)
- Minimum 100 examples per property test (`@settings(max_examples=100)`)
- Training scripts (`train_vae.py`, `train_corruption_classifier.py`, `train_diffusion.py`) are standalone — run them in order before executing `main.py`
- The `checkpoints/` directory must exist before training; create it with `mkdir -p checkpoints`
