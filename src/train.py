"""
train.py
────────────────────────────────────────────────────────────────
Loop de treinamento para segmentação de tumor cerebral (LGG MRI).

Cada execução cria uma nova versão (v1, v2, v3 …) sem sobrescrever
runs anteriores. Os arquivos gerados seguem o padrão:
    checkpoints/<modelo>_<backbone>_v<N>_best.pth
    checkpoints/<modelo>_<backbone>_v<N>_history.csv

Uso básico:
    python src/train.py

Uso com argumentos:
    python src/train.py --model unetpp --backbone efficientnet-b4 --epochs 80
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as A
from dataset import get_dataloaders, TRAIN_TRANSFORMS
from models import MODEL_MAP, build_model
from metrics import dice_coefficient, iou_score


# ─────────────────────────────────────────────────────────────────
# Versionamento de runs
# ─────────────────────────────────────────────────────────────────

def next_run_version(ckpt_dir: Path, model: str, backbone: str) -> str:
    """
    Retorna o próximo sufixo de versão (_v1, _v2, …) para um par
    (model, backbone), garantindo que nenhum checkpoint existente
    seja sobrescrito.
    """
    prefix = f"{model}_{backbone}_v"
    existing = [
        p for p in ckpt_dir.glob(f"{prefix}*_best.pth")
    ]
    used = set()
    for p in existing:
        # extrai o número entre 'v' e '_best'
        stem = p.stem  # ex: unet_resnet34_v2_best
        try:
            n = int(stem.replace(f"{model}_{backbone}_v", "").replace("_best", ""))
            used.add(n)
        except ValueError:
            pass
    version = 1
    while version in used:
        version += 1
    return f"v{version}"


# ─────────────────────────────────────────────────────────────────
# Configurações padrão
# ─────────────────────────────────────────────────────────────────

DEFAULTS = {
    "dataset_dir":        "dataset",
    "checkpoints":        "checkpoints",
    "model":              "unetpp",
    "backbone":           "efficientnet-b2",
    "epochs":             100,
    "batch_size":         16,
    "lr":                 1e-4,
    "patience":           10,
    "num_workers":        4,
    "amp":                True,
    "dropout":            0.3,
    "weight_decay":       1e-3,
    "encoder_lr_factor":  0.1,   # encoder usa lr * fator; decoder usa lr completo
}


# ─────────────────────────────────────────────────────────────────
# Um epoch de treino / validação
# ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool, scaler=None):
    model.train() if training else model.eval()

    total_loss = 0.0
    total_dice, total_iou, n_dice = 0.0, 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                preds = model(images)
                loss  = criterion(preds, masks)

            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                # treino: batch-level (rápido, não precisa de precisão por amostra)
                total_dice += dice_coefficient(preds, masks)
                total_iou  += iou_score(preds, masks)
                n_dice += 1
            else:
                # validação: per-sample apenas em fatias com tumor
                # torna val_dice comparável ao test_dice (with_tumor) da inferência
                preds_bin = (torch.sigmoid(preds) > 0.5).float()
                for pred_s, mask_s in zip(preds_bin, masks):
                    if mask_s.sum() > 0:
                        inter = (pred_s * mask_s).sum()
                        total_dice += (2.0 * inter / (pred_s.sum() + mask_s.sum() + 1e-7)).item()
                        total_iou  += (inter / (pred_s.sum() + mask_s.sum() - inter + 1e-7)).item()
                        n_dice += 1

            total_loss += loss.item()

    avg_dice = total_dice / n_dice if n_dice > 0 else 0.0
    avg_iou  = total_iou  / n_dice if n_dice > 0 else 0.0
    return total_loss / len(loader), avg_dice, avg_iou


# ─────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        cfg["dataset_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"Batches — train: {len(train_loader)} | val: {len(val_loader)}")

    model = build_model(cfg["model"], cfg["backbone"], dropout=cfg["dropout"]).to(device)
    print(f"Modelo: {cfg['model']} + {cfg['backbone']} | dropout={cfg['dropout']} | weight_decay={cfg['weight_decay']} | AMP={cfg['amp']} | encoder_lr_factor={cfg['encoder_lr_factor']}\n")

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss  = nn.BCEWithLogitsLoss()
    criterion = lambda preds, masks: 0.5 * dice_loss(preds, masks) + 0.5 * bce_loss(preds, masks)

    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    optimizer = Adam(
        [
            {"params": encoder_params, "lr": cfg["lr"] * cfg["encoder_lr_factor"]},
            {"params": decoder_params, "lr": cfg["lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    use_amp = cfg["amp"] and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir  = Path(cfg["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_ver   = next_run_version(ckpt_dir, cfg["model"], cfg["backbone"])
    run_stem  = f"{cfg['model']}_{cfg['backbone']}_{run_ver}"
    ckpt_path = ckpt_dir / f"{run_stem}_best.pth"
    print(f"Run: {run_stem}  →  {ckpt_path.name}")

    best_dice     = 0.0
    epochs_no_imp = 0
    history       = []

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Dice':>10} | {'Val Loss':>9} | {'Val Dice':>9} | {'Val IoU':>8} | {'LR':>8}")
    print("-" * 80)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = run_epoch(model, train_loader, criterion, optimizer, device, training=True,  scaler=scaler)
        val_loss,   val_dice,   val_iou   = run_epoch(model, val_loader,   criterion, optimizer, device, training=False, scaler=None)

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[1]["lr"]   # LR do decoder (referência)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss, "train_dice": train_dice, "train_iou": train_iou,
            "val_loss":   val_loss,   "val_dice":   val_dice,   "val_iou":   val_iou,
            "lr":         current_lr,
        })

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_dice:>10.4f} | {val_loss:>9.4f} | {val_dice:>9.4f} | {val_iou:>8.4f} | {current_lr:>8.2e}  ({elapsed:.0f}s)")

        if val_dice > best_dice:
            best_dice     = val_dice
            epochs_no_imp = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice":        val_dice,
                "val_iou":         val_iou,
                "cfg":             cfg,
                "train_transforms": A.to_dict(TRAIN_TRANSFORMS),
            }, ckpt_path)
            print(f"         ✓ checkpoint salvo (val_dice={val_dice:.4f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= cfg["patience"]:
                print(f"\nEarly stopping na época {epoch} (sem melhora por {cfg['patience']} épocas)")
                break

    history_path = ckpt_dir / f"{run_stem}_history.csv"
    if history:
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"Histórico salvo em:  {history_path}")

    print(f"\nTreino concluído. Melhor val_dice: {best_dice:.4f}")
    print(f"Checkpoint salvo em: {ckpt_path}")
    return history, ckpt_path


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Treino de segmentação LGG MRI")
    p.add_argument("--dataset_dir",  default=DEFAULTS["dataset_dir"])
    p.add_argument("--checkpoints",  default=DEFAULTS["checkpoints"])
    p.add_argument("--model",        default=DEFAULTS["model"],       choices=list(MODEL_MAP))
    p.add_argument("--backbone",     default=DEFAULTS["backbone"])
    p.add_argument("--epochs",       default=DEFAULTS["epochs"],      type=int)
    p.add_argument("--batch_size",   default=DEFAULTS["batch_size"],  type=int)
    p.add_argument("--lr",           default=DEFAULTS["lr"],          type=float)
    p.add_argument("--patience",     default=DEFAULTS["patience"],    type=int)
    p.add_argument("--num_workers",  default=DEFAULTS["num_workers"], type=int)
    p.add_argument("--amp",          default=DEFAULTS["amp"],          action=argparse.BooleanOptionalAction)
    p.add_argument("--dropout",           default=DEFAULTS["dropout"],           type=float)
    p.add_argument("--weight_decay",      default=DEFAULTS["weight_decay"],      type=float)
    p.add_argument("--encoder_lr_factor", default=DEFAULTS["encoder_lr_factor"], type=float)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
