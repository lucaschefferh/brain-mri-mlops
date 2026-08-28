"""
split_dataset.py
────────────────────────────────────────────────────────────────
Divide o dataset LGG-MRI em train / val / test (70 / 15 / 15 %).
A divisão é feita por PACIENTE para evitar data leakage.

Estrutura esperada:
    archive/kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif
            TCGA_CS_4941_19960909_1_mask.tif
            ...
        ...

Estrutura gerada:
    dataset/
        train/
            images/  *.tif
            masks/   *_mask.tif
        val/
            images/
            masks/
        test/
            images/
            masks/
"""

import random
import shutil
from pathlib import Path

SOURCE_DIR  = Path("archive/kaggle_3m")
OUTPUT_DIR  = Path("dataset")
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios devem somar 1"


def copy_patient(patient_dir: Path, images_out: Path, masks_out: Path):
    """Copia todas as imagens e máscaras de um paciente para as pastas de saída."""
    for f in sorted(patient_dir.glob("*.tif")):
        dest = masks_out if "_mask" in f.name else images_out
        shutil.copy2(f, dest / f.name)


def split_patients(patient_dirs: list[Path]) -> dict[str, list[Path]]:
    """Divide lista de pacientes em train/val/test sem sobreposição."""
    n_total = len(patient_dirs)
    n_train = round(n_total * TRAIN_RATIO)
    n_val   = round(n_total * VAL_RATIO)

    return {
        "train": patient_dirs[:n_train],
        "val":   patient_dirs[n_train : n_train + n_val],
        "test":  patient_dirs[n_train + n_val :],
    }


def main():
    random.seed(RANDOM_SEED)

    patient_dirs = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])
    random.shuffle(patient_dirs)
    splits = split_patients(patient_dirs)

    n_total = len(patient_dirs)
    print(f"\nTotal de pacientes: {n_total}")
    for name, patients in splits.items():
        print(f"  {name:>5} : {len(patients)} pacientes")
    print()

    for split_name, patients in splits.items():
        images_dir = OUTPUT_DIR / split_name / "images"
        masks_dir  = OUTPUT_DIR / split_name / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        n_slices = 0
        for patient in patients:
            copy_patient(patient, images_dir, masks_dir)
            n_slices += len(list(patient.glob("*.tif"))) // 2

        print(f"[{split_name:>5}] {len(patients):>3} pacientes | {n_slices:>4} slices copiados")

    print(f"\nDataset salvo em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
