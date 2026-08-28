# Brain MRI Segmentation — MLOps

Pipeline de MLOps sobre um modelo de segmentação de glioma de baixo grau (LGG) em MRI:
reprodutibilidade, quantização para edge, serving na AWS, observabilidade e análise de trade-off
entre inferência na nuvem e no dispositivo.

> **Em construção.** O modelo e os experimentos de origem estão em
> [FastCamp / Visão Computacional / Card 7](https://github.com/lucaschefferh/FastCamp).
> Este repositório continua aquele trabalho a partir do checkpoint treinado.

> Não é um dispositivo médico e não deve orientar decisão clínica.
> Dados: [LGG Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (TCGA, público e desidentificado).

## Modelo de partida

| | |
|---|---|
| Arquitetura | UNet++ / EfficientNet-B2 |
| Parâmetros | 10,47 M |
| Peso fp32 | 41,9 MB |
| Val Dice | 0,778 |
| Recall (fatias com tumor) | 0,819 |
| Gap treino/validação | 0,040 |

## Status das fases

- [ ] **00** Base verificável — ambiente travado, baseline reproduzido, harness e testes
- [ ] **01** Export ONNX e teste de paridade
- [ ] **02** Quantização e portão de recall
- [ ] **03** Medição em dispositivo real
- [ ] **04** Serving na AWS
- [ ] **05** CI/CD com portões de qualidade
- [ ] **06** Observabilidade sem ground truth
- [ ] **07** Drift entre instituições e retreino
- [ ] **08** Relatório de trade-off
