"""Fixtures compartilhadas: caminhos e parsing de nomes de arquivo."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
SPLITS = ("train", "val", "test")


def patient_id(filename: str) -> str:
    """
    Extrai o identificador do paciente de um nome de arquivo do LGG dataset.

        TCGA_CS_4941_19960909_10.tif  ->  TCGA_CS_4941_19960909

    Os quatro primeiros campos identificam o exame de um paciente; o último é
    o índice da fatia. Duas fatias do mesmo paciente NÃO podem cair em splits
    diferentes, sob pena de vazamento.
    """
    return "_".join(Path(filename).stem.split("_")[:4])


def institution(filename: str) -> str:
    """Código da instituição de origem (segundo campo): CS, DU, EZ, FG, HT."""
    return Path(filename).stem.split("_")[1]


@pytest.fixture(scope="session")
def split_images() -> dict[str, list[Path]]:
    """Mapeia cada split para a lista de imagens que ele contém."""
    if not DATASET_DIR.exists():
        pytest.skip(f"dataset não encontrado em {DATASET_DIR}")
    images = {s: sorted((DATASET_DIR / s / "images").glob("*.tif")) for s in SPLITS}
    for split, paths in images.items():
        if not paths:
            pytest.skip(f"split '{split}' está vazio")
    return images
