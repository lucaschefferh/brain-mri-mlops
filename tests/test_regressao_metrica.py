"""
Regressão de métrica: o modelo de referência ainda entrega o que prometeu?

Este é o teste que dá autoridade ao pipeline — é ele que, na Fase 05, vai
barrar um deploy quando uma variante quantizada degradar além do aceitável.

Roda em duas velocidades:
    • rápido  — subconjunto determinístico de 40 fatias, para o CI de cada push
    • lento   — test set completo, comparado contra results/baseline.json
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASELINE = REPO / "results" / "baseline.json"
CHECKPOINT = REPO / "checkpoints" / "unetpp_efficientnet-b2_v1_best.pth"

# O baseline guarda a média em precisão cheia, então a tolerância só precisa
# cobrir não-determinismo de hardware — não ruído de formatação.
TOL = 1e-9


def _requisitos():
    if not BASELINE.exists():
        pytest.skip("results/baseline.json ausente — rode src/make_baseline.py")
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint ausente: {CHECKPOINT}")
    pytest.importorskip("torch")


@pytest.fixture(scope="module")
def baseline():
    _requisitos()
    return json.loads(BASELINE.read_text())


def _avaliar(indices=None):
    from dataset import VAL_TRANSFORMS, BrainMRIDataset
    from evaluate import evaluate, resumir
    from predictors import TorchPredictor

    ds = BrainMRIDataset(REPO / "dataset" / "test", transform=VAL_TRANSFORMS)
    if indices is not None:
        ds.image_paths = [ds.image_paths[i] for i in indices]
        ds.mask_paths = [ds.mask_paths[i] for i in indices]

    df = evaluate(TorchPredictor(CHECKPOINT), ds, thresholds=(0.5,), progress=False)
    return resumir(df).iloc[0]


def test_subconjunto_produz_metricas_validas(baseline):
    """Rápido: 40 fatias espaçadas uniformemente, cobrindo casos com e sem tumor.

    Não compara contra o baseline (a média de 40 fatias não é a de 589) — checa
    que o pipeline roda inteiro e devolve métricas plausíveis. É o teste que
    pega cano quebrado: modelo que não carrega, contrato de shape violado,
    pré-processamento fora do lugar.
    """
    _requisitos()
    got = _avaliar(indices=list(range(0, 589, 15)))

    assert got.n_slices == 40
    assert got.n_com_tumor > 0, "o subconjunto precisa conter fatias com tumor"
    assert 0.0 <= got.dice_tumor <= 1.0
    assert got.dice_tumor > 0.5, f"queda grosseira de qualidade: dice={got.dice_tumor:.4f}"


@pytest.mark.slow
def test_test_set_completo_reproduz_o_baseline(baseline):
    """Lento: as 589 fatias contra o baseline congelado, sem tolerância folgada."""
    esperado = baseline["metricas_teste"]
    got = _avaliar()

    assert int(got.n_slices) == esperado["n_slices"]
    assert int(got.n_com_tumor) == esperado["n_slices_com_tumor"]

    for chave in ("dice_tumor", "iou_tumor", "precision_tumor", "recall_tumor",
                  "acuracia_deteccao"):
        delta = abs(float(got[chave]) - esperado[chave])
        assert delta < TOL, (
            f"{chave} divergiu do baseline: {got[chave]:.9f} vs {esperado[chave]:.9f} "
            f"(Δ={delta:.2e}, tolerância={TOL:.0e})"
        )
