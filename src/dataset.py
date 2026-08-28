"""
dataset.py
────────────────────────────────────────────────────────────────
Classe BrainMRIDataset + transforms de treino e validação.

Estrutura esperada em data_dir:
    dataset/
        train/
            images/   TCGA_*_N.tif
            masks/    TCGA_*_N_mask.tif
        val/
            images/
            masks/
        test/
            images/
            masks/
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────

# Usados apenas no treino
# Augmentações selecionadas para MRI cerebral clínico:
#   - Flips e rotações leves simulam variações de posicionamento do paciente
#   - GaussNoise simula ruído térmico natural do scanner MRI
#   - GaussianBlur simula variações de resolução entre equipamentos
#   - ElasticTransform/GridDistortion simulam variações anatômicas sutis
#   - CoarseDropout com buracos pequenos (≤32px) simula artefatos de movimento e oclusão parcial
TRAIN_TRANSFORMS = A.Compose([
    # Geométricas — variações reais de posicionamento
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        scale=(0.85, 1.15),
        rotate=(-20, 20),
        p=0.5,
    ),
    # Deformação anatômica suave
    A.ElasticTransform(alpha=60, sigma=8, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
    # Zoom leve
    A.RandomResizedCrop(
        size=(256, 256),
        scale=(0.85, 1.0),
        ratio=(1.0, 1.0),
        p=0.4,
    ),
    # Intensidade — variações de contraste e ruído de scanner
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussNoise(std_range=(0.04, 0.20), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Usados em val e test — apenas normalização
VAL_TRANSFORMS = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class BrainMRIDataset(Dataset):
    """
    Carrega pares (imagem, máscara) a partir de uma pasta de split.

    Parâmetros
    ----------
    split_dir : str | Path
        Caminho para a pasta do split, ex: "dataset/train".
        Deve conter subpastas `images/` e `masks/`.
    transform : albumentations.Compose | None
        Pipeline de augmentation/normalização. Se None, retorna
        arrays numpy brutos (útil para inspeção visual).
    """

    def __init__(self, split_dir: str | Path, transform=None):
        self.images_dir = Path(split_dir) / "images"
        self.masks_dir  = Path(split_dir) / "masks"
        self.transform  = transform

        # Lista todas as imagens e monta os pares
        self.image_paths = sorted(self.images_dir.glob("*.tif"))
        self.mask_paths  = [
            self.masks_dir / (p.stem + "_mask.tif")
            for p in self.image_paths
        ]

        # Verificação de integridade: todas as máscaras devem existir
        missing = [m for m in self.mask_paths if not m.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} máscara(s) não encontrada(s). Exemplo: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Leitura
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # (H, W, 3) uint8

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)  # (H, W) uint8
        mask = (mask > 0).astype(np.uint8)                      # garante binário 0/1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # tensor float (3, H, W)
            mask  = augmented["mask"]    # tensor uint8 (H, W)

            # Adiciona dimensão de canal na máscara: (1, H, W) float
            mask = mask.unsqueeze(0).float()

        return image, mask

    # ── Utilitários ──────────────────────────────────────────────

    def class_balance(self) -> dict:
        """Retorna a proporção de fatias com e sem tumor no split."""
        positive = 0
        for mask_path in self.mask_paths:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m.max() > 0:
                positive += 1
        total = len(self.mask_paths)
        return {
            "total": total,
            "with_tumor": positive,
            "without_tumor": total - positive,
            "positive_ratio": round(positive / total, 3),
        }


# ─────────────────────────────────────────────────────────────────
# Funções auxiliares de construção
# ─────────────────────────────────────────────────────────────────

def get_datasets(dataset_dir: str | Path):
    """
    Retorna os três splits prontos com os transforms corretos.

    Uso:
        train_ds, val_ds, test_ds = get_datasets("dataset")
    """
    dataset_dir = Path(dataset_dir)
    train_ds = BrainMRIDataset(dataset_dir / "train", transform=TRAIN_TRANSFORMS)
    val_ds   = BrainMRIDataset(dataset_dir / "val",   transform=VAL_TRANSFORMS)
    test_ds  = BrainMRIDataset(dataset_dir / "test",  transform=VAL_TRANSFORMS)
    return train_ds, val_ds, test_ds


def get_dataloaders(dataset_dir: str | Path, batch_size: int = 16, num_workers: int = 4):
    """
    Retorna DataLoaders prontos para treino.

    Uso:
        train_loader, val_loader, test_loader = get_dataloaders("dataset")
    """
    train_ds, val_ds, test_ds = get_datasets(dataset_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
