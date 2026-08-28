"""
predictors.py
────────────────────────────────────────────────────────────────
Cada predictor sabe rodar UM tipo de modelo e nada mais.

O contrato é o mesmo para todos — é ele que torna a comparação entre
variantes honesta, porque garante que a única coisa diferente entre
elas é o motor de inferência:

    entrada : np.ndarray (B, 3, H, W) float32, já normalizado
    saída   : np.ndarray (B, 1, H, W) float32, PROBABILIDADES em [0, 1]

Devolver probabilidade (e não logit, nem máscara binária) é deliberado:
    • o threshold vira decisão do harness, num único lugar do código;
    • dá para varrer vários pontos de operação sem reexecutar o modelo;
    • esconde a diferença de onde o sigmoid vive — no PyTorch ele é
      aplicado aqui, mas no ONNX ele pode acabar fundido no grafo.
"""

from pathlib import Path
from typing import Protocol

import numpy as np


class Predictor(Protocol):
    """Qualquer coisa que respeite este formato serve como predictor."""

    name: str

    def __call__(self, batch: np.ndarray) -> np.ndarray: ...


class TorchPredictor:
    """Roda um checkpoint .pth do projeto. Reconstrói a arquitetura a partir
    do `cfg` salvo dentro do próprio checkpoint, então não é preciso informar
    modelo e backbone por fora."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import torch

        from models import build_model

        self._torch = torch
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model(
            cfg["model"], cfg["backbone"], dropout=cfg.get("dropout", 0.0), pretrained=False
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.name = f"torch:{cfg['model']}_{cfg['backbone']}"
        self.meta = {
            "runtime": "pytorch",
            "device": str(self.device),
            "modelo": cfg["model"],
            "backbone": cfg["backbone"],
            "epoca": ckpt.get("epoch"),
            "val_dice": ckpt.get("val_dice"),
        }

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x))
        return probs.cpu().numpy()
