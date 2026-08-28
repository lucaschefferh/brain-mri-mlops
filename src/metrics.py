"""
metrics.py
────────────────────────────────────────────────────────────────
Funções de métricas compartilhadas entre train.py e inference.py.

  dice_coefficient   — Dice por batch (tensores PyTorch, treino/val)
  iou_score          — IoU por batch  (tensores PyTorch, treino/val)
  per_sample_metrics — Dice, IoU, precisão e recall por amostra (numpy, inferência)
"""

import numpy as np
import torch


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Dice calculado sobre o batch inteiro, após binarização."""
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    return (2.0 * intersection / (preds.sum() + targets.sum() + 1e-7)).item()


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """IoU calculado sobre o batch inteiro, após binarização."""
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection / (union + 1e-7)).item()


def per_sample_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Métricas pixel-a-pixel para um único par (pred, target) binários."""
    tp = int((pred * target).sum())
    fp = int((pred * (1 - target)).sum())
    fn = int(((1 - pred) * target).sum())
    tn = int(((1 - pred) * (1 - target)).sum())

    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou       = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)

    return {
        "dice":      round(dice,      6),
        "iou":       round(iou,       6),
        "precision": round(precision, 6),
        "recall":    round(recall,    6),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
    }
