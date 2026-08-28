"""
make_baseline.py
────────────────────────────────────────────────────────────────
Congela o desempenho de referência de um checkpoint em results/baseline.json.

O arquivo registra PROCEDÊNCIA, não só números: hash do checkpoint, commit do
código e versões do ambiente. Daqui em diante toda variante (ONNX, fp16, int8,
dispositivo) se compara contra este registro — não contra um número no README.

Uso:
    python src/make_baseline.py --predictions results/unetpp_efficientnet-b2_v1
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def env_versions() -> dict:
    import cv2
    import numpy as np
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def summarize(df: pd.DataFrame) -> dict:
    """Agregados sobre o test set. Fatias sem tumor têm Dice 0 por definição
    matemática, então as métricas de qualidade usam apenas fatias com tumor."""
    tumor = df[df.has_tumor_gt == 1]
    return {
        "n_slices": int(len(df)),
        "n_slices_com_tumor": int(len(tumor)),
        "dice_tumor": round(float(tumor.dice.mean()), 6),
        "iou_tumor": round(float(tumor.iou.mean()), 6),
        "precision_tumor": round(float(tumor.precision.mean()), 6),
        "recall_tumor": round(float(tumor.recall.mean()), 6),
        "acuracia_deteccao": round(float((df.has_tumor_gt == df.has_tumor_pred).mean()), 6),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Congela o baseline de um run de inferência")
    p.add_argument("--predictions", required=True, help="Pasta results/<run> gerada pela inferência")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--output", default="results/baseline.json")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    run_dir = Path(args.predictions)
    info = json.loads((run_dir / "model_info.json").read_text())
    ckpt = Path(args.checkpoints) / f"{run_dir.name}_best.pth"

    baseline = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run_dir.name,
        "modelo": {**info, "threshold": args.threshold},
        "checkpoint": {
            "arquivo": ckpt.name,
            "sha256": sha256(ckpt),
            "bytes": ckpt.stat().st_size,
        },
        "codigo": {"commit": git_commit()},
        "ambiente": env_versions(),
        "metricas_teste": summarize(pd.read_csv(run_dir / "test_predictions.csv")),
    }

    out = Path(args.output)
    out.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n")
    print(f"baseline gravado em {out}")
    print(json.dumps(baseline["metricas_teste"], indent=2))


if __name__ == "__main__":
    main()
