"""
inference.py
────────────────────────────────────────────────────────────────
Carrega um checkpoint, roda inferência na base de teste e salva
predições em results/<stem_do_checkpoint>/.

O nome da pasta de saída é derivado do próprio checkpoint,
preservando runs anteriores sem sobrescrever:
    unet_resnet34_v1_best.pth  →  results/unet_resnet34_v1/
    unet_resnet34_v2_best.pth  →  results/unet_resnet34_v2/

Saídas geradas:
    test_predictions.csv  — métricas por amostra
    pred_masks.npy        — máscaras preditas, shape (N, H, W) uint8
    model_info.json       — metadados do checkpoint

Uso:
    python src/inference.py
    python src/inference.py --checkpoint checkpoints/unet_resnet34_v1_best.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import VAL_TRANSFORMS, BrainMRIDataset
from metrics import per_sample_metrics
from models import build_model

# ─────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────

def find_best_checkpoint(checkpoints_dir: Path) -> Path:
    """Retorna o *_best.pth com maior val_dice na pasta de checkpoints."""
    candidates = sorted(checkpoints_dir.glob("*_best.pth"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum checkpoint encontrado em '{checkpoints_dir}'")

    print("Checkpoints disponíveis:")
    best_path, best_dice = None, -1.0
    for p in candidates:
        ckpt  = torch.load(p, map_location="cpu", weights_only=False)
        dice  = ckpt.get("val_dice", -1.0)
        iou   = ckpt.get("val_iou", float("nan"))
        epoch = ckpt.get("epoch", "?")
        print(f"  {p.name:40s}  val_dice={dice:.4f}  val_iou={iou:.4f}  época={epoch}")
        if dice > best_dice:
            best_dice, best_path = dice, p

    print(f"\nSelecionado: {best_path.name} (val_dice={best_dice:.4f})\n")
    return best_path


# ─────────────────────────────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Reconstrói o modelo a partir do checkpoint e carrega os pesos salvos."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = build_model(
        cfg["model"],
        cfg["backbone"],
        dropout=cfg.get("dropout", 0.0),
        pretrained=False,              # pesos vêm do checkpoint
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    dataset: BrainMRIDataset,
    device: torch.device,
    batch_size: int = 16,
    threshold: float = 0.5,
    num_workers: int = 4,
) -> tuple[list[dict], np.ndarray]:
    """
    Roda inferência em todo o dataset.

    Retorna:
        records    — lista de dicts com métricas por amostra
        pred_masks — array (N, H, W) uint8 com máscaras preditas binárias
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    records, pred_masks = [], []
    sample_idx = 0

    for images, masks in tqdm(loader, desc="Inferência"):
        images = images.to(device)

        with torch.no_grad():
            preds_bin = (torch.sigmoid(model(images)) > threshold).float()

        preds_np   = preds_bin.cpu().numpy()   # (B, 1, H, W)
        targets_np = masks.cpu().numpy()        # (B, 1, H, W)

        for i in range(preds_np.shape[0]):
            pred   = preds_np[i, 0]
            target = targets_np[i, 0]
            records.append({
                "image_name":     dataset.image_paths[sample_idx].name,
                "has_tumor_gt":   int(target.max() > 0),
                "has_tumor_pred": int(pred.max() > 0),
                **per_sample_metrics(pred, target),
            })
            pred_masks.append(pred.astype(np.uint8))
            sample_idx += 1

    return records, np.stack(pred_masks)  # (N, H, W) uint8


# ─────────────────────────────────────────────────────────────────
# Persistência
# ─────────────────────────────────────────────────────────────────

def save_results(
    out_dir: Path,
    records: list[dict],
    pred_masks: np.ndarray,
    ckpt: dict,
):
    """Salva CSV de predições, array de máscaras e metadados do modelo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ckpt["cfg"]

    pd.DataFrame(records).to_csv(out_dir / "test_predictions.csv", index=False)
    np.save(out_dir / "pred_masks.npy", pred_masks)
    (out_dir / "model_info.json").write_text(json.dumps({
        "model":    cfg["model"],
        "backbone": cfg["backbone"],
        "epoch":    ckpt["epoch"],
        "val_dice": ckpt["val_dice"],
        "val_iou":  ckpt["val_iou"],
    }, indent=2))

    print(f"\nResultados salvos em: {out_dir.resolve()}")
    print(f"  test_predictions.csv  ({len(records)} amostras)")
    print(f"  pred_masks.npy        {pred_masks.shape}")
    print("  model_info.json")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inferência na base de teste — LGG MRI")
    p.add_argument("--checkpoint",   default=None,          help="Caminho do checkpoint .pth (padrão: melhor em --checkpoints)")
    p.add_argument("--checkpoints",  default="checkpoints", help="Pasta onde buscar checkpoints")
    p.add_argument("--dataset_dir",  default="dataset")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--batch_size",   default=16,  type=int)
    p.add_argument("--threshold",    default=0.5, type=float, help="Limiar de binarização (sigmoid)")
    p.add_argument("--num_workers",  default=4,   type=int)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(Path(args.checkpoints))
    model, ckpt = load_model(ckpt_path, device)
    cfg = ckpt["cfg"]
    print(f"Modelo: {cfg['model']} + {cfg['backbone']}  (época {ckpt['epoch']}, val_dice={ckpt['val_dice']:.4f})\n")

    test_ds = BrainMRIDataset(Path(args.dataset_dir) / "test", transform=VAL_TRANSFORMS)
    print(f"Amostras no conjunto de teste: {len(test_ds)}\n")

    records, pred_masks = run_inference(
        model, test_ds, device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_workers=args.num_workers,
    )

    # Deriva o nome da pasta de saída do stem do checkpoint
    # Ex: unet_resnet34_v2_best.pth  →  results/unet_resnet34_v2/
    run_name = ckpt_path.stem.replace("_best", "")  # unet_resnet34_v2
    out_dir  = Path(args.results_dir) / run_name
    save_results(out_dir, records, pred_masks, ckpt)


if __name__ == "__main__":
    main()
