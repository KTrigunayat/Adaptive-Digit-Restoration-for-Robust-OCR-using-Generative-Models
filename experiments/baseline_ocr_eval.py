"""
baseline_ocr_eval.py
Standalone script to benchmark the OCR classifier on:
  1. Clean MNIST images (a_clean)
  2. MNIST images corrupted with each of the three corruption types independently
     (a_corrupted per type: gaussian_noise, motion_blur, spatial_masking)

Results are written as JSON to the path specified in cfg["evaluation"]["output_path"].

Run independently — no generative models required.
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from data.distortion_engine import apply_distortion
from src.models.ocr_classifier import OCRClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.metrics import compute_ocr_accuracy

log = get_logger("baseline_ocr_eval")

CORRUPTION_TYPES = ["gaussian_noise", "motion_blur", "spatial_masking"]
BATCH_SIZE = 256


def build_corrupted_dataset(
    clean_images: torch.Tensor,
    labels: torch.Tensor,
    corruption_type: str,
    seed: int = 42,
) -> TensorDataset:
    """Apply a single corruption type to all images and return a TensorDataset.

    Args:
        clean_images: Float32 tensor of shape (N, 1, 28, 28) with values in [0, 1].
        labels:       Integer label tensor of shape (N,).
        corruption_type: One of the three canonical corruption type strings.
        seed: Base random seed; each image uses seed + index for reproducibility.

    Returns:
        TensorDataset of (corrupted_images, labels) where corrupted_images is
        float32 in [0, 1] with shape (N, 1, 28, 28).
    """
    corrupted_list = []
    for idx, img_tensor in enumerate(clean_images):
        # Convert float32 [0,1] tensor → uint8 numpy array for distortion engine
        img_np = (img_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        corrupted_np = apply_distortion(img_np, corruption_type, seed=seed + idx)
        # Convert back to float32 [0,1] tensor with channel dim
        corrupted_float = torch.from_numpy(corrupted_np.astype(np.float32) / 255.0)
        if corrupted_float.ndim == 2:
            corrupted_float = corrupted_float.unsqueeze(0)  # (1, 28, 28)
        corrupted_list.append(corrupted_float)

    corrupted_tensor = torch.stack(corrupted_list)  # (N, 1, 28, 28)
    return TensorDataset(corrupted_tensor, labels)


def run_baseline_eval(cfg: dict | None = None) -> dict:
    """Run baseline OCR evaluation on clean and corrupted MNIST test images.

    Args:
        cfg: Configuration dict. If None, loads from config.yaml.

    Returns:
        Results dict with keys:
          - "a_clean": float — OCR accuracy on clean images
          - "a_corrupted": dict — {corruption_type: accuracy} for each type
    """
    if cfg is None:
        cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.get("seed", 42)

    # ------------------------------------------------------------------
    # Load MNIST test set
    # ------------------------------------------------------------------
    transform = transforms.Compose([transforms.ToTensor()])
    raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")
    test_set = datasets.MNIST(
        root=raw_dir, train=False, download=True, transform=transform
    )

    # Collect all test images and labels into tensors for reuse
    all_images = torch.stack([img for img, _ in test_set])   # (N, 1, 28, 28)
    all_labels = torch.tensor([lbl for _, lbl in test_set])  # (N,)

    # ------------------------------------------------------------------
    # Load OCR model
    # ------------------------------------------------------------------
    model = OCRClassifier().to(device)
    ocr_checkpoint = cfg.get("ocr", {}).get("checkpoint", "checkpoints/ocr.pth")
    try:
        model.load_state_dict(
            torch.load(ocr_checkpoint, map_location=device)
        )
        log.info(f"Loaded OCR weights from {ocr_checkpoint}.")
    except FileNotFoundError:
        log.warning(
            f"No pretrained weights found at {ocr_checkpoint} — "
            "evaluating with random init (expect ~10% accuracy)."
        )

    model.eval()

    # ------------------------------------------------------------------
    # Compute a_clean
    # ------------------------------------------------------------------
    clean_loader = DataLoader(
        TensorDataset(all_images, all_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    a_clean = _eval_loader(model, clean_loader, device)
    log.info(f"a_clean (clean MNIST): {a_clean * 100:.2f}%")

    # ------------------------------------------------------------------
    # Compute a_corrupted for each corruption type
    # ------------------------------------------------------------------
    a_corrupted: dict[str, float] = {}

    for corruption_type in CORRUPTION_TYPES:
        log.info(f"Building corrupted dataset for: {corruption_type} ...")
        corrupted_ds = build_corrupted_dataset(
            all_images, all_labels, corruption_type, seed=seed
        )
        corrupted_loader = DataLoader(
            corrupted_ds, batch_size=BATCH_SIZE, shuffle=False
        )
        acc = _eval_loader(model, corrupted_loader, device)
        a_corrupted[corruption_type] = acc
        log.info(f"a_corrupted[{corruption_type}]: {acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # Assemble results and write to JSON
    # ------------------------------------------------------------------
    results = {
        "a_clean": a_clean,
        "a_corrupted": a_corrupted,
    }

    output_path = Path(cfg.get("evaluation", {}).get("output_path", "experiments/results/eval_report.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Baseline results written to {output_path}")

    return results


def _eval_loader(model: OCRClassifier, loader: DataLoader, device: str) -> float:
    """Evaluate OCR accuracy over a DataLoader using compute_ocr_accuracy.

    Accumulates predictions across all batches and returns overall accuracy.
    """
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model.predict(images)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Use compute_ocr_accuracy from metrics.py for the final computation
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    cfg = load_config()
    results = run_baseline_eval(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
