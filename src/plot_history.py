"""
plot_history.py
────────────────────────────────────────────────────────────────
Gera gráficos das curvas de treinamento a partir dos CSVs de histórico.

Uso:
    python src/plot_history.py
    python src/plot_history.py --checkpoints_dir checkpoints --output_dir results
"""

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "train": "#2196F3",
    "val":   "#F44336",
}


def plot_model_history(df: pd.DataFrame, model_label: str, out_path: Path):
    best_epoch = df.loc[df["val_dice"].idxmax(), "epoch"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Curvas de Treinamento — {model_label}", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    ax_loss, ax_dice, ax_iou, ax_lr = axes

    def _plot(ax, train_col, val_col, title, ylabel):
        has_train = train_col in df.columns
        has_val   = val_col   in df.columns
        if not has_train and not has_val:
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.axis("off")
            return
        if has_train:
            ax.plot(df["epoch"], df[train_col], color=COLORS["train"], label="Treino", linewidth=1.8)
        if has_val:
            ax.plot(df["epoch"], df[val_col],   color=COLORS["val"],   label="Validação", linewidth=1.8)
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Melhor época ({int(best_epoch)})")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Época", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(ax_loss, "train_loss", "val_loss", "Loss (Dice + BCE)", "Loss")
    _plot(ax_dice, "train_dice", "val_dice", "Dice Coefficient",  "Dice")
    _plot(ax_iou,  "train_iou",  "val_iou",  "IoU Coefficient",   "IoU")

    # Learning rate
    ax_lr.plot(df["epoch"], df["lr"], color="#4CAF50", linewidth=1.8)
    ax_lr.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Melhor época ({int(best_epoch)})")
    ax_lr.set_title("Learning Rate", fontsize=10, fontweight="bold")
    ax_lr.set_xlabel("Época", fontsize=9)
    ax_lr.set_ylabel("LR", fontsize=9)
    ax_lr.set_yscale("log")
    ax_lr.legend(fontsize=8)
    ax_lr.grid(True, alpha=0.3)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out_path}")


def main(checkpoints_dir: Path, output_dir: Path):
    csv_files = sorted(checkpoints_dir.glob("*_history.csv"))
    if not csv_files:
        print(f"Nenhum arquivo *_history.csv encontrado em '{checkpoints_dir}'")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        model_label = csv_path.stem.replace("_history", "").replace("_", " / ", 1)
        df = pd.read_csv(csv_path)
        out_path = output_dir / (csv_path.stem.replace("_history", "_training_curves") + ".png")
        plot_model_history(df, model_label, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plota curvas de treinamento dos CSVs de histórico")
    p.add_argument("--checkpoints_dir", default="checkpoints", type=Path)
    p.add_argument("--output_dir",      default="results",     type=Path)
    args = p.parse_args()
    main(args.checkpoints_dir, args.output_dir)
