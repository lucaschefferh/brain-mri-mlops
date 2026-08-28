"""
evaluate.py
────────────────────────────────────────────────────────────────
Harness de avaliação: um único caminho de medição para TODAS as variantes
do modelo (PyTorch, ONNX, fp16, int8, dispositivo).

Por que isso existe
───────────────────
A comparação entre variantes é o produto deste projeto. Se cada variante
tivesse seu próprio laço de avaliação, as cópias divergiriam em silêncio —
um threshold diferente aqui, uma normalização esquecida ali — e os números
deixariam de ser comparáveis sem que nada quebrasse.

Aqui o que varia (como rodar o modelo) entra por parâmetro; o que não varia
(batching, métricas, contabilidade) mora num lugar só.

    evaluate(TorchPredictor("checkpoints/..._best.pth"), dataset)
    evaluate(OnnxPredictor("models/..._int8.onnx"),      dataset)

Varredura de threshold
──────────────────────
`thresholds` aceita vários valores e todos são calculados numa única passada
pelo modelo. Isso torna a escolha do ponto de operação barata: quantizar
desloca a distribuição de saída, então o 0.5 ótimo do fp32 raramente é o
ótimo do int8, e comparar variantes em thresholds fixos pode ser injusto.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from metrics import per_sample_metrics  # noqa: E402


def evaluate(
    predictor,
    dataset,
    batch_size: int = 16,
    thresholds: tuple[float, ...] = (0.5,),
    num_workers: int = 4,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Roda `predictor` sobre `dataset` e devolve métricas por amostra.

    Parâmetros
    ----------
    predictor : callable
        (B, 3, H, W) float32 numpy -> (B, 1, H, W) float32 com probabilidades.
    dataset : BrainMRIDataset
        Precisa usar VAL_TRANSFORMS — o mesmo pré-processamento do treino.
    batch_size : int
        Afeta só a velocidade da avaliação, não as métricas. Para MEDIR
        LATÊNCIA use batch_size=1, que é o cenário real de serving e mobile.
    thresholds : tuple[float, ...]
        Pontos de operação a avaliar. Todos saem de uma única passada.

    Retorna
    -------
    DataFrame em formato longo: uma linha por (amostra, threshold).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,      # a ordem precisa casar com dataset.image_paths
        num_workers=num_workers,
        pin_memory=True,
    )

    registros: list[dict] = []
    idx = 0
    tempo_modelo = 0.0

    iterador = tqdm(loader, desc="Avaliação", disable=not progress)
    for imagens, mascaras in iterador:
        # Contrato em numpy: o predictor não deve precisar saber o que é tensor.
        lote = imagens.numpy().astype(np.float32, copy=False)

        t0 = time.perf_counter()
        probs = predictor(lote)                      # (B, 1, H, W) probabilidades
        tempo_modelo += time.perf_counter() - t0

        alvos = mascaras.numpy()                     # (B, 1, H, W)

        for i in range(probs.shape[0]):
            prob = probs[i, 0]
            alvo = alvos[i, 0].astype(np.uint8)
            nome = dataset.image_paths[idx].name

            for t in thresholds:
                pred = (prob > t).astype(np.uint8)
                registros.append({
                    "image_name": nome,
                    "threshold": t,
                    "has_tumor_gt": int(alvo.max() > 0),
                    "has_tumor_pred": int(pred.max() > 0),
                    **per_sample_metrics(pred, alvo),
                })
            idx += 1

    df = pd.DataFrame(registros)
    df.attrs["tempo_modelo_s"] = round(tempo_modelo, 4)
    df.attrs["n_amostras"] = idx
    df.attrs["predictor"] = getattr(predictor, "name", type(predictor).__name__)
    return df


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o resultado de `evaluate` por threshold.

    Fatias sem tumor têm Dice 0 por definição matemática, mesmo quando a
    predição está correta (vazio contra vazio). Por isso as métricas de
    qualidade usam só fatias com tumor, e a capacidade de dizer "não há
    tumor aqui" é medida à parte, pela acurácia de detecção.
    """
    linhas = []
    for t, g in df.groupby("threshold"):
        com_tumor = g[g.has_tumor_gt == 1]
        linhas.append({
            "threshold": t,
            "n_slices": len(g),
            "n_com_tumor": len(com_tumor),
            "dice_tumor": com_tumor.dice.mean(),
            "iou_tumor": com_tumor.iou.mean(),
            "precision_tumor": com_tumor.precision.mean(),
            "recall_tumor": com_tumor.recall.mean(),
            "acuracia_deteccao": (g.has_tumor_gt == g.has_tumor_pred).mean(),
        })
    return pd.DataFrame(linhas).sort_values("threshold").reset_index(drop=True)
