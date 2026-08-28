"""
analysis.py
────────────────────────────────────────────────────────────────
Lê as predições salvas pelo inference.py e gera:
  - test_summary.csv          resumo estatístico por subconjunto
  - metrics_distribution.png  histogramas de Dice e IoU
  - best_predictions.png      grade das melhores predições
  - worst_predictions.png     grade das piores predições

Deve ser executado após inference.py.

Uso:
    python src/analysis.py
    python src/analysis.py --predictions results/unet_resnet34_v1
    python src/analysis.py --predictions results/unet_resnet34_v2
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dataset import VAL_TRANSFORMS, BrainMRIDataset

# ─────────────────────────────────────────────────────────────────
# Carregamento de dados
# ─────────────────────────────────────────────────────────────────

def find_results_dir(results_root: Path) -> Path:
    """
    Retorna a pasta de resultados dentro de results_root.
    Se houver apenas uma, a retorna automaticamente.
    Se houver múltiplas, imprime a lista e retorna a mais recente
    (última em ordem alfabética), sugerindo --predictions para escolher.
    """
    candidates = sorted([
        d for d in results_root.iterdir()
        if d.is_dir() and (d / "test_predictions.csv").exists()
    ])
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum resultado encontrado em '{results_root}'. Execute inference.py primeiro."
        )
    if len(candidates) > 1:
        nomes = [c.name for c in candidates]
        print(f"Múltiplas pastas de resultados encontradas: {nomes}")
        print(f"Usando a mais recente: {candidates[-1].name}")
        print("Use --predictions para especificar outra.\n")
    return candidates[-1]


def load_predictions(model_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Carrega CSV de predições, máscaras e metadados do modelo."""
    df         = pd.read_csv(model_dir / "test_predictions.csv")
    pred_masks = np.load(model_dir / "pred_masks.npy")          # (N, H, W) uint8

    info_path  = model_dir / "model_info.json"
    model_info = json.loads(info_path.read_text()) if info_path.exists() else {}

    return df, pred_masks, model_info


# ─────────────────────────────────────────────────────────────────
# Estatísticas
# ─────────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame, model_info: dict) -> pd.DataFrame:
    """Resumo estatístico por subconjunto (all / with_tumor / without_tumor)."""
    metrics = ["dice", "iou", "precision", "recall"]
    rows = []

    subsets = [
        ("all",           pd.Series([True] * len(df))),
        ("with_tumor",    df["has_tumor_gt"] == 1),
        ("without_tumor", df["has_tumor_gt"] == 0),
    ]

    for label, mask in subsets:
        sub = df.loc[mask, metrics]
        row = {"subset": label, "n_samples": int(mask.sum())}
        for m in metrics:
            row[f"{m}_mean"]   = round(sub[m].mean(),   6)
            row[f"{m}_std"]    = round(sub[m].std(),    6)
            row[f"{m}_median"] = round(sub[m].median(), 6)
            row[f"{m}_min"]    = round(sub[m].min(),    6)
            row[f"{m}_max"]    = round(sub[m].max(),    6)
        rows.append(row)

    rows[0]["tumor_detection_accuracy"] = round(
        (df["has_tumor_gt"] == df["has_tumor_pred"]).mean(), 6
    )

    if model_info:
        rows[0].update({
            "checkpoint_model":    model_info.get("model"),
            "checkpoint_backbone": model_info.get("backbone"),
            "checkpoint_epoch":    model_info.get("epoch"),
            "val_dice":            model_info.get("val_dice"),
            "val_iou":             model_info.get("val_iou"),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Visualizações
# ─────────────────────────────────────────────────────────────────

def _load_image_and_gt(dataset: BrainMRIDataset, idx: int) -> tuple[np.ndarray, np.ndarray]:
    img_path  = dataset.image_paths[idx]
    mask_path = dataset.masks_dir / (img_path.stem + "_mask.tif")
    img  = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) * 255
    return img, mask


def save_prediction_grid(
    title: str,
    indices: list[int],
    records: list[dict],
    pred_masks: np.ndarray,
    dataset: BrainMRIDataset,
    out_path: Path,
):
    """Salva grade N×3: imagem original | ground truth | predição."""
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, label in enumerate(["Imagem original", "Ground Truth", "Predição"]):
        axes[0, col].set_title(label, fontsize=9, fontweight="bold")

    for row, idx in enumerate(indices):
        rec  = records[idx]
        img, gt = _load_image_and_gt(dataset, idx)
        pred = pred_masks[idx] * 255

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(
            f"{rec['image_name']}\nDice={rec['dice']:.3f}  IoU={rec['iou']:.3f}",
            fontsize=7, rotation=0, labelpad=130, va="center",
        )
        axes[row, 1].imshow(gt,   cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=255)
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_metrics_histogram(df: pd.DataFrame, out_path: Path):
    """Salva histogramas de Dice e IoU separados por presença de tumor."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric in zip(axes, ["dice", "iou"], strict=True):
        with_tumor    = df.loc[df["has_tumor_gt"] == 1, metric]
        without_tumor = df.loc[df["has_tumor_gt"] == 0, metric]
        mean          = df[metric].mean()

        ax.hist(with_tumor,    bins=30, alpha=0.6, label="Com tumor",    color="tomato")
        ax.hist(without_tumor, bins=30, alpha=0.6, label="Sem tumor",    color="steelblue")
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.5,
                   label=f"Média geral={mean:.3f}")
        ax.set_xlabel(metric.upper(), fontsize=11)
        ax.set_ylabel("Nº amostras", fontsize=10)
        ax.set_title(f"Distribuição {metric.upper()} — base de teste", fontsize=11)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# Saída no terminal
# ─────────────────────────────────────────────────────────────────

def print_summary(summary_df: pd.DataFrame, model_info: dict):
    label = (
        f"{model_info['model']} + {model_info['backbone']}"
        if model_info else "modelo desconhecido"
    )
    all_row = summary_df.loc[summary_df["subset"] == "all"].iloc[0]

    print("\n" + "=" * 60)
    print(f"RESULTADOS NO CONJUNTO DE TESTE ({label})")
    print("=" * 60)
    for metric in ["dice", "iou", "precision", "recall"]:
        print(
            f"  {metric.upper():12s} "
            f"média={all_row[f'{metric}_mean']:.4f}  "
            f"std={all_row[f'{metric}_std']:.4f}  "
            f"mediana={all_row[f'{metric}_median']:.4f}"
        )
    tumor_acc = all_row.get("tumor_detection_accuracy", float("nan"))
    print(f"  {'Detecção':12s} acurácia={tumor_acc:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Análise e visualização — LGG MRI")
    p.add_argument("--predictions", default=None,    help="Pasta com resultados do modelo (padrão: busca automática em --results_dir)")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--n_vis",       default=12, type=int, help="Amostras por grade visual")
    return p.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.predictions) if args.predictions else find_results_dir(Path(args.results_dir))
    print(f"Carregando predições de: {model_dir}\n")

    df, pred_masks, model_info = load_predictions(model_dir)
    records = df.to_dict("records")
    print(f"Amostras: {len(df)}")

    dataset = BrainMRIDataset(Path(args.dataset_dir) / "test", transform=VAL_TRANSFORMS)

    # ── Resumo estatístico ───────────────────────────────────────
    summary_df = build_summary(df, model_info)
    summary_df.to_csv(model_dir / "test_summary.csv", index=False)
    print(f"Resumo salvo:       {model_dir / 'test_summary.csv'}")

    # ── Histogramas ──────────────────────────────────────────────
    save_metrics_histogram(df, model_dir / "metrics_distribution.png")
    print(f"Histograma salvo:   {model_dir / 'metrics_distribution.png'}")

    # ── Grades de predições ──────────────────────────────────────
    n_vis = min(args.n_vis, len(records))
    with_tumor_idx = [i for i, r in enumerate(records) if r["has_tumor_gt"] == 1]

    worst_idx = sorted(with_tumor_idx, key=lambda i: records[i]["dice"])[:n_vis]
    if worst_idx:
        save_prediction_grid(
            f"Piores predições — {n_vis} menores Dice (fatias com tumor)",
            worst_idx, records, pred_masks, dataset,
            model_dir / "worst_predictions.png",
        )
        print(f"Grade (pior) salva: {model_dir / 'worst_predictions.png'}")

    best_idx = sorted(with_tumor_idx, key=lambda i: records[i]["dice"], reverse=True)[:n_vis]
    if best_idx:
        save_prediction_grid(
            f"Melhores predições — {n_vis} maiores Dice (fatias com tumor)",
            best_idx, records, pred_masks, dataset,
            model_dir / "best_predictions.png",
        )
        print(f"Grade (melhor) salva: {model_dir / 'best_predictions.png'}")

    # ── Resumo terminal ──────────────────────────────────────────
    print_summary(summary_df, model_info)
    print(f"\nTodos os arquivos em: {model_dir.resolve()}")


if __name__ == "__main__":
    main()
