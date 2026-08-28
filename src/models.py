"""
models.py
────────────────────────────────────────────────────────────────
Registro central de modelos de segmentação.
Importado por train.py e inference.py para evitar duplicação.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

MODEL_MAP = {
    "unet":       smp.Unet,
    "unetpp":     smp.UnetPlusPlus,
    "deeplabv3p": smp.DeepLabV3Plus,
}


def build_model(
    model_name: str,
    backbone: str,
    dropout: float = 0.0,
    pretrained: bool = True,
) -> nn.Module:
    cls = MODEL_MAP.get(model_name)
    if cls is None:
        raise ValueError(f"Modelo '{model_name}' desconhecido. Opções: {list(MODEL_MAP)}")
    return cls(
        encoder_name=backbone,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,      # logits brutos; sigmoid aplicado externamente
        decoder_dropout=dropout,
    )
