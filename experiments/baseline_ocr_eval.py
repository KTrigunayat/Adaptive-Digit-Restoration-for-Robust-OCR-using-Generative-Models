"""
baseline_ocr_eval.py
Standalone script to benchmark the OCR classifier on:
  1. Clean MNIST images
  2. Corrupted images (noise / blur / mask)
  3. DIP-preprocessed images

Run independently — no generative models required.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.ocr_classifier import OCRClassifier
from src.utils.logger import get_logger

log = get_logger("baseline_ocr_eval")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256


def evaluate(model: OCRClassifier, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model.predict(images)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root="data/raw", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = OCRClassifier().to(DEVICE)

    # Load pretrained weights if available
    try:
        model.load_state_dict(torch.load("experiments/ocr_weights.pth", map_location=DEVICE))
        log.info("Loaded pretrained OCR weights.")
    except FileNotFoundError:
        log.warning("No pretrained weights found — evaluating with random init (expect ~10% accuracy).")

    acc = evaluate(model, test_loader)
    log.info(f"Clean MNIST accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
