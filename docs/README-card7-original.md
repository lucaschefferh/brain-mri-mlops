# Brain MRI Segmentation — LGG Tumor

Segmentação semântica de tumores cerebrais (Low-Grade Glioma) em imagens de MRI, usando PyTorch e `segmentation-models-pytorch`. O projeto cobre o pipeline completo: divisão do dataset, treinamento, inferência e análise quantitativa/visual dos resultados, com quatro iterações de treino documentadas em [`Relatório 7 - Lucas Scheffer.pdf`](Relatório%207%20-%20Lucas%20Scheffer.pdf).

## Vídeo Pitch

📽️ **[Assista ao pitch do projeto (3 min)](videoDocumentacao.mp4)**

Contextualização → problema → solução → principais diferenciais → resultados obtidos.

> A apresentação completa, usada na defesa de 8 minutos, está em [`Apresentação - Segmentação Tumor Cerebral LGG-MRI.pdf`](Apresentação%20-%20Segmentação%20Tumor%20Cerebral%20LGG-MRI.pdf).

## Dataset

[LGG Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) — 110 pacientes do TCGA, imagens FLAIR em `.tif` com máscaras binárias de segmentação (3929 pares imagem/máscara, ~35% com tumor / 65% sem tumor).

```
archive/kaggle_3m/
    TCGA_<instituição>_<paciente>_<slice>.tif
    TCGA_<instituição>_<paciente>_<slice>_mask.tif
    data.csv
```

## Estrutura do Projeto

```
.
├── src/
│   ├── split_dataset.py   # divide o dataset por paciente (70/15/15 %)
│   ├── dataset.py         # BrainMRIDataset, transforms, dataloaders
│   ├── models.py          # registro central de arquiteturas (build_model)
│   ├── metrics.py         # Dice/IoU (batch) e métricas por amostra
│   ├── train.py           # loop de treinamento com early stopping e versionamento de runs
│   ├── inference.py       # inferência no conjunto de teste (métricas + máscaras preditas)
│   ├── analysis.py        # resumo estatístico e visualizações a partir da inferência
│   └── plot_history.py    # curvas de treino a partir dos CSVs de histórico
├── archive/kaggle_3m/     # dataset original (não versionado)
├── dataset/               # splits gerados por split_dataset.py (não versionado)
│   ├── train/images+masks
│   ├── val/images+masks
│   └── test/images+masks
├── checkpoints/           # melhores pesos e histórico CSV por modelo/run (não versionado)
├── results/               # métricas, CSVs e grades visuais por modelo/run
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt
```

> **Nota:** as versões no `requirements.txt` foram especificadas manualmente e podem não existir. Se `pip install` falhar, instale sem versões fixas ou gere um `requirements.txt` limpo com `pip freeze` após instalar manualmente.

## Como Usar

### 1. Dividir o dataset

```bash
python src/split_dataset.py
```

Divide por **paciente** (evita data leakage) em 70 % treino / 15 % validação / 15 % teste. Gera a pasta `dataset/` com subpastas `train/`, `val/` e `test/`.

### 2. Treinar

```bash
# Configuração padrão: UNet++ + EfficientNet-B2, 100 épocas
python src/train.py

# Personalizado
python src/train.py --model unet --backbone resnet34 --epochs 50 --batch_size 8
```

| Argumento               | Padrão              | Opções / Descrição                                    |
| ----------------------- | ------------------- | ------------------------------------------------------ |
| `--model`               | `unetpp`             | `unet`, `unetpp`, `deeplabv3p`                          |
| `--backbone`            | `efficientnet-b2`    | qualquer encoder suportado pelo `segmentation-models-pytorch` |
| `--epochs`              | `100`                | —                                                        |
| `--batch_size`          | `16`                 | —                                                        |
| `--lr`                  | `1e-4`               | learning rate do decoder                                 |
| `--patience`            | `10`                 | early stopping                                           |
| `--num_workers`         | `4`                  | —                                                        |
| `--amp` / `--no-amp`    | `True`               | precisão mista (menos VRAM, mais velocidade)             |
| `--dropout`             | `0.3`                | dropout no decoder                                       |
| `--weight_decay`        | `1e-3`               | regularização L2 (Adam)                                  |
| `--encoder_lr_factor`   | `0.1`                | fator multiplicador do lr aplicado ao encoder pré-treinado (preserva conhecimento do ImageNet) |

Cada execução gera uma **versão nova** sem sobrescrever runs anteriores: `checkpoints/<modelo>_<backbone>_v<N>_best.pth` e `checkpoints/<modelo>_<backbone>_v<N>_history.csv`.

### 3. Inferência no conjunto de teste

```bash
# Seleciona automaticamente o checkpoint com maior val_dice
python src/inference.py

# Checkpoint específico
python src/inference.py --checkpoint checkpoints/unetpp_efficientnet-b2_v1_best.pth
```

Gera em `results/<modelo>_<backbone>_v<N>/`:

- `test_predictions.csv` — métricas por amostra (Dice, IoU, Precision, Recall)
- `pred_masks.npy` — máscaras preditas binárias (não versionado — regenerável a partir do checkpoint)
- `model_info.json` — metadados do checkpoint (época, val_dice, val_iou)

### 4. Análise e visualização

```bash
python src/analysis.py --predictions results/unetpp_efficientnet-b2_v1
```

A partir da saída da inferência, gera no mesmo diretório:

- `test_summary.csv` — resumo estatístico por subconjunto (todas / com tumor / sem tumor)
- `metrics_distribution.png` — histogramas de Dice e IoU
- `best_predictions.png` / `worst_predictions.png` — grades com as melhores/piores segmentações (fatias com tumor)

### 5. Curvas de treino

```bash
python src/plot_history.py --output_dir results/unetpp_efficientnet-b2_v1
```

Lê os CSVs de histórico em `checkpoints/` e gera `<modelo>_<backbone>_v<N>_training_curves.png` (loss, Dice, IoU e learning rate por época).

## Resultados

Foram treinadas quatro iterações, testando arquitetura, tamanho de encoder e regularização para controlar o overfitting observado nos primeiros treinos:

| Modelo             | Backbone         | Melhor época | Val Dice   | Gap treino/val | Dice (tumor) | IoU (tumor) | Recall (tumor) | Detecção |
| ------------------- | ---------------- | :-----------: | :--------: | :-------------: | :-----------: | :----------: | :-------------: | :-------: |
| UNet                | ResNet34          | 19            | 0.612      | 0.282           | 0.760         | 0.676        | 0.774            | 92.7 %    |
| UNet++              | EfficientNet-B4   | 55            | 0.603      | 0.340           | **0.763**     | **0.686**    | 0.765            | **95.3 %** |
| UNet++              | EfficientNet-B4 (regularizado) | 18 | 0.606      | 0.281           | 0.755         | 0.674        | 0.765            | 95.1 %    |
| UNet                | ResNet18          | 35            | 0.751      | 0.082           | 0.681         | 0.603        | 0.692            | 90.3 %    |
| UNet++              | EfficientNet-B0   | 35            | 0.775      | 0.059           | 0.722         | 0.639        | 0.768            | 89.0 %    |
| **UNet++**          | **EfficientNet-B2** | **32**      | **0.778**  | **0.040**       | 0.743         | 0.656        | **0.819**        | 88.3 %    |

> Dados completos por modelo em [`results/comparacao_modelos.csv`](results/comparacao_modelos.csv). O Dice "geral" reportado pela inferência (~0.27–0.29 em todos os modelos) é enganoso: inclui as fatias sem tumor, onde o Dice é zero por definição matemática mesmo com predição correta. As colunas acima usam apenas as **fatias com tumor presente**.

**Melhor modelo: UNet++ / EfficientNet-B2.** Não é o de Dice/IoU absolutamente mais alto — os encoders maiores (ResNet34, EfficientNet-B4) chegam a 0.75–0.76 — mas apresenta de longe o menor gap treino/validação do experimento (0.04 contra 0.28–0.34) e o maior recall entre os seis modelos (0.82), a métrica mais relevante em contexto clínico (menor chance de deixar passar um tumor). O detalhamento de cada iteração, as causas do overfitting e a justificativa das escolhas estão no relatório final.

### Exemplos de segmentação

Predições do melhor modelo (UNet++ / EfficientNet-B2) no conjunto de teste — 2 dos melhores casos e 2 dos piores, lado a lado com a imagem original e o ground truth:

![Exemplos de segmentação: imagem original, ground truth e predição](results/unetpp_efficientnet-b2_v1/sample_predictions.png)

Nos dois melhores casos (tumores grandes e bem definidos) a predição praticamente coincide com o ground truth. Nos dois piores casos, ambos têm tumores muito pequenos (poucas dezenas de pixels) — o modelo captou parcialmente um e não detectou o outro. Essa é a principal limitação observada: **tumores pequenos são consistentemente mais difíceis de segmentar** do que tumores grandes, independentemente do modelo.

Distribuição de Dice e IoU no conjunto de teste completo, separando fatias com e sem tumor:

![Distribuição de Dice e IoU no conjunto de teste](results/unetpp_efficientnet-b2_v1/metrics_distribution.png)

## Arquitetura e Treinamento

- **Modelos:** U-Net, U-Net++ e DeepLabV3+ via `segmentation-models-pytorch`
- **Encoders testados:** ResNet18/34, EfficientNet-B0/B2/B4 — pré-treinados no ImageNet
- **Loss:** 0.5 × DiceLoss + 0.5 × BCEWithLogitsLoss
- **Otimizador:** Adam com learning rate diferenciado (encoder × `encoder_lr_factor`, decoder no `lr` completo), weight decay e `ReduceLROnPlateau` (modo max, fator 0.5, patience 5)
- **Precisão mista (AMP):** reduz uso de VRAM e acelera o treino
- **Augmentations (treino):** flip horizontal/vertical, rotação/translação/escala (`Affine`), transformação elástica e distorção em grade, zoom leve (`RandomResizedCrop`), brilho/contraste/gama, ruído gaussiano e blur leve, `CoarseDropout`
- **Regularização:** dropout no decoder e weight decay, ajustados por iteração para conter o overfitting
- **Métricas:** Dice coefficient, IoU, precisão e recall (calculados após sigmoid + binarização em 0.5)
