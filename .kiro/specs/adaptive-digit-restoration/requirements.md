# Requirements Document

## Introduction

The Adaptive Digit Restoration system is a hybrid AI pipeline that restores corrupted MNIST handwritten digit images to improve downstream OCR accuracy. It combines classical Digital Image Processing (DIP) techniques with a Variational Autoencoder (VAE) and a Conditional Latent Diffusion Model (LDM) to adaptively denoise images corrupted by Gaussian noise, motion blur, or spatial masking. The system is evaluated by comparing OCR accuracy across clean, corrupted, and restored images.

---

## Glossary

- **System**: The complete Adaptive Digit Restoration pipeline.
- **Distortion_Engine**: The module responsible for applying synthetic corruptions to clean MNIST images.
- **DIP_Layer**: The Digital Image Processing preprocessing stage (median filter + histogram equalization).
- **VAE**: The Variational Autoencoder that encodes images into a 64-dimensional latent space and decodes them back.
- **Corruption_Classifier**: A lightweight classifier that predicts the corruption type from a preprocessed image.
- **Diffusion_Engine**: The Latent Diffusion Model (LDM) that performs stepwise denoising in latent space, conditioned on corruption type.
- **UNet**: The U-Net backbone used by the Diffusion_Engine for noise prediction.
- **OCR_Classifier**: The pre-trained CNN used to evaluate digit recognition accuracy on clean, corrupted, and restored images.
- **Latent_Vector**: A 64-dimensional floating-point tensor produced by the VAE encoder representing an image in latent space.
- **Corruption_Type**: One of three discrete categories — `gaussian_noise`, `motion_blur`, or `spatial_masking`.
- **ELBO**: Evidence Lower Bound — the VAE training objective combining reconstruction loss and KL divergence.
- **PSNR**: Peak Signal-to-Noise Ratio — a pixel-level image quality metric in decibels.
- **OCR_Accuracy**: The fraction of digit images correctly classified by the OCR_Classifier.
- **Restored_Image**: A 1×28×28 grayscale tensor produced by decoding a denoised Latent_Vector through the VAE decoder.
- **Clean_Image**: An unmodified MNIST image tensor of shape 1×28×28 with pixel values in [0, 1].

---

## Requirements

### Requirement 1: Synthetic Corruption Generation

**User Story:** As a researcher, I want to generate reproducible synthetic corruptions of MNIST digits, so that I can train and evaluate the restoration pipeline under controlled degradation conditions.

#### Acceptance Criteria

1. WHEN a Clean_Image and a Corruption_Type of `gaussian_noise` are provided, THE Distortion_Engine SHALL apply additive Gaussian noise with a standard deviation σ drawn uniformly from [0.1, 0.5].
2. WHEN a Clean_Image and a Corruption_Type of `motion_blur` are provided, THE Distortion_Engine SHALL convolve the image with a motion blur kernel of size 3×3 or 5×5.
3. WHEN a Clean_Image and a Corruption_Type of `spatial_masking` are provided, THE Distortion_Engine SHALL zero out a randomly positioned contiguous 8×8 pixel region.
4. THE Distortion_Engine SHALL return a corrupted image as a NumPy array with the same shape as the input (1×28×28 or 28×28).
5. IF an unrecognized Corruption_Type is provided, THEN THE Distortion_Engine SHALL raise a `ValueError` identifying the invalid type and listing valid options.
6. THE Distortion_Engine SHALL accept a random seed parameter to enable reproducible corruption generation.

---

### Requirement 2: DIP Preprocessing Layer

**User Story:** As a researcher, I want corrupted images to be stabilized by classical filters before entering the generative model, so that the VAE receives cleaner inputs and training is more stable.

#### Acceptance Criteria

1. WHEN a corrupted image is provided, THE DIP_Layer SHALL apply a median filter with kernel size 3 to suppress salt-and-pepper noise.
2. WHEN a corrupted image is provided, THE DIP_Layer SHALL apply global histogram equalization to normalize pixel intensity contrast.
3. THE DIP_Layer SHALL apply the median filter before histogram equalization in the preprocessing sequence.
4. THE DIP_Layer SHALL return a preprocessed image as a NumPy array with the same spatial dimensions as the input.
5. THE DIP_Layer SHALL operate on CPU and accept uint8 or float32 grayscale arrays.

---

### Requirement 3: Variational Autoencoder (VAE)

**User Story:** As a researcher, I want a VAE trained on clean MNIST images, so that the system can encode digit images into a structured latent space suitable for diffusion-based denoising.

#### Acceptance Criteria

1. THE VAE SHALL encode a 1×28×28 input tensor into a mean vector μ and a log-variance vector log σ, each of dimension 64.
2. THE VAE SHALL use exactly 3 convolutional layers with stride 2 in the encoder to progressively downsample the spatial dimensions.
3. THE VAE SHALL use the reparameterization trick to sample a Latent_Vector z = μ + ε·exp(0.5·log σ), where ε ~ N(0, I).
4. THE VAE SHALL decode a 64-dimensional Latent_Vector back to a 1×28×28 tensor with pixel values in [0, 1].
5. THE VAE SHALL compute training loss as L = ReconstructionLoss + β·KL-Divergence, where ReconstructionLoss is binary cross-entropy summed over all pixels.
6. WHEN trained on clean MNIST images using the Adam optimizer with learning rate 1e-4, THE VAE SHALL achieve a reconstruction loss indicating high-fidelity output (PSNR ≥ 20 dB on the MNIST test set).
7. THE VAE encoder SHALL be freezable (gradient computation disabled) for use during Diffusion_Engine training.

---

### Requirement 4: Corruption Type Classifier

**User Story:** As a researcher, I want the system to automatically identify the type of corruption present in an input image, so that the Diffusion_Engine can be conditioned on the correct corruption type for adaptive restoration.

#### Acceptance Criteria

1. THE Corruption_Classifier SHALL accept a preprocessed 1×28×28 image tensor as input and output a probability distribution over the three Corruption_Types.
2. THE Corruption_Classifier SHALL produce a 1-hot encoded vector of dimension 3 representing the predicted Corruption_Type for use as a conditioning signal.
3. WHEN evaluated on a held-out set of synthetically corrupted MNIST images, THE Corruption_Classifier SHALL achieve a classification accuracy of at least 85%.
4. IF the input image does not match any known Corruption_Type pattern, THEN THE Corruption_Classifier SHALL return the class with the highest predicted probability without raising an error.

---

### Requirement 5: Conditional Latent Diffusion Model

**User Story:** As a researcher, I want a diffusion model that denoises VAE latent vectors conditioned on corruption type, so that the restoration is adaptive to the specific degradation present in each image.

#### Acceptance Criteria

1. THE Diffusion_Engine SHALL implement a DDPM-style forward process that adds Gaussian noise to a Latent_Vector over T timesteps using a linear noise schedule with β₁ = 1e-4 and β_T = 0.02.
2. WHEN performing the reverse process, THE Diffusion_Engine SHALL use the UNet to predict the noise at each timestep, conditioned on the 1-hot encoded Corruption_Type vector.
3. THE UNet SHALL accept as input the concatenation of the noisy Latent_Vector and the Corruption_Type conditioning vector at each denoising step.
4. THE Diffusion_Engine SHALL train using MSE loss between the predicted noise and the actual noise added during the forward process.
5. WHEN the Diffusion_Engine is trained, THE VAE encoder SHALL remain frozen (no gradient updates).
6. WHEN the reverse process completes, THE Diffusion_Engine SHALL return a denoised Latent_Vector of dimension 64 suitable for decoding by the VAE decoder.
7. THE Diffusion_Engine SHALL support a configurable number of timesteps T, defaulting to 1000.

---

### Requirement 6: End-to-End Restoration Pipeline

**User Story:** As a researcher, I want a single orchestrated pipeline that takes a corrupted image and produces a restored image, so that I can evaluate the full system in a reproducible way.

#### Acceptance Criteria

1. WHEN a corrupted MNIST image is provided, THE System SHALL execute the following stages in order: DIP_Layer preprocessing → VAE encoding → Corruption_Classifier prediction → Diffusion_Engine reverse process → VAE decoding.
2. THE System SHALL accept a batch of images as input and produce a batch of Restored_Images of the same shape (B×1×28×28).
3. THE System SHALL load model weights from paths specified in a configuration file (config.yaml).
4. IF a required model checkpoint file is not found at the configured path, THEN THE System SHALL raise a `FileNotFoundError` with the missing path.
5. THE System SHALL support execution on both CPU and CUDA devices, selecting CUDA automatically when available and falling back to CPU otherwise.
6. THE System SHALL log each pipeline stage name and the shape of its output tensor at INFO level.

---

### Requirement 7: OCR Evaluation and Metrics

**User Story:** As a researcher, I want to measure OCR accuracy and image quality metrics across clean, corrupted, and restored images, so that I can quantify the benefit of the restoration pipeline.

#### Acceptance Criteria

1. THE OCR_Classifier SHALL be trained exclusively on clean MNIST training images and SHALL NOT be updated during restoration training.
2. WHEN evaluating a batch of images, THE System SHALL compute OCR_Accuracy separately for clean images (A_clean), corrupted images (A_corrupted), and Restored_Images (A_restored).
3. THE System SHALL compute PSNR between each Restored_Image and its corresponding Clean_Image.
4. THE System SHALL compute the ELBO on the VAE reconstruction of clean images as a measure of latent space quality.
5. WHEN evaluation is complete, THE System SHALL produce a summary report containing A_clean, A_corrupted, A_restored, mean PSNR, and mean ELBO.
6. THE System SHALL write the evaluation summary report to a file path specified in the configuration.

---

### Requirement 8: Baseline Performance Measurement

**User Story:** As a researcher, I want to measure OCR accuracy on corrupted images before any restoration, so that I can establish a performance baseline to compare against the restored results.

#### Acceptance Criteria

1. WHEN the baseline evaluation script is executed, THE OCR_Classifier SHALL be evaluated on MNIST test images corrupted with each of the three Corruption_Types independently.
2. THE System SHALL record A_corrupted for each Corruption_Type separately (gaussian_noise, motion_blur, spatial_masking).
3. THE System SHALL record A_clean as the OCR_Classifier accuracy on unmodified MNIST test images.
4. THE System SHALL write baseline results to a structured output file (JSON or CSV) at a path specified in the configuration.

---

### Requirement 9: Configuration and Reproducibility

**User Story:** As a researcher, I want all hyperparameters and file paths to be centrally configurable, so that experiments are reproducible and easy to modify without changing source code.

#### Acceptance Criteria

1. THE System SHALL read all hyperparameters (latent dimension, learning rate, timesteps T, β values, batch size) from config.yaml.
2. THE System SHALL read all data and checkpoint file paths from config.yaml.
3. WHEN a required configuration key is missing, THE System SHALL raise a descriptive `KeyError` identifying the missing key.
4. THE System SHALL accept a random seed value from config.yaml and apply it to PyTorch, NumPy, and Python's random module at startup to ensure reproducibility.

---

### Requirement 10: Corruption Round-Trip Fidelity

**User Story:** As a researcher, I want to verify that the restoration pipeline produces images that are structurally closer to the clean originals than the corrupted inputs, so that I can confirm the system is genuinely improving image quality.

#### Acceptance Criteria

1. FOR ALL Corruption_Types, THE System SHALL produce Restored_Images with a mean PSNR strictly greater than the mean PSNR of the corresponding corrupted images relative to the Clean_Images.
2. FOR ALL Corruption_Types, THE System SHALL produce A_restored strictly greater than A_corrupted when evaluated on the MNIST test set.
3. THE System SHALL preserve the digit class label through the restoration process — the OCR_Classifier prediction on a Restored_Image SHALL match the ground-truth label at a rate equal to A_restored.
