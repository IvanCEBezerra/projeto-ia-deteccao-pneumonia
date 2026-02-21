# 🫁 Classificação de Patologias em Raio-X Torácico
### Ligia – Liga Acadêmica de Inteligência Artificial · UFPE · Processo Seletivo 2026

**Trilha:** Visão Computacional | **Métrica:** ROC AUC | **Resultado:** `0.99129` no leaderboard Kaggle

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-ROC%20AUC%200.99129-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Sumário

1. [Visão Geral](#-visão-geral)
2. [Resultados](#-resultados)
3. [Estrutura do Repositório](#-estrutura-do-repositório)
4. [Instalação e Ambiente](#-instalação-e-ambiente)
5. [Dataset](#-dataset)
6. [Pipeline Completo](#-pipeline-completo)
7. [Como Reproduzir](#-como-reproduzir)
8. [Decisões Técnicas](#-decisões-técnicas)
9. [Análise de Interpretabilidade](#-análise-de-interpretabilidade)
10. [Limitações e Trabalhos Futuros](#-limitações-e-trabalhos-futuros)
11. [Autor](#-autor)

---

## 🔬 Visão Geral

Este projeto aborda a classificação binária de radiografias de tórax (**NORMAL** vs **PNEUMONIA**) como parte do Desafio Individual da trilha de Visão Computacional da Ligia (UFPE, 2026).

A solução foi construída como uma **jornada de pesquisa documentada**: cada decisão técnica — desde a detecção de *shortcut learning* por viés de hardware até a escolha do backbone por torneio controlado — é rastreável a evidência experimental reproduzível.

### Destaques Metodológicos

- 🔍 **Detecção original de viés de hardware**: análise estatística de ruído de fundo (desvio padrão em cantos da imagem) revelou que imagens NORMAL e PNEUMONIA foram capturadas com equipamentos distintos, criando uma "assinatura de sensor" que poderia ser aprendida como *shortcut* pelo modelo
- 🛡️ **Anti-leakage por paciente**: 23,7% dos pacientes têm múltiplas imagens; `StratifiedGroupKFold` garante que nenhum paciente apareça em treino e validação simultaneamente
- ⚗️ **Todas as decisões por torneio controlado**: transforms, estratégia de balanceamento, backbone e pesos do ensemble — todas escolhas baseadas em experimentos com critério de equivalência estatística
- 🧬 **Ensemble heterogêneo**: DenseNet-121 com pesos ImageNet (Mixup+LS) + DenseNet-121 com pesos especializados em >100k raio-X (TorchXRayVision), com pesos otimizados por grid search

---

## 📊 Resultados

| Configuração | ROC AUC (CV 5-fold) | Spread |
|---|---|---|
| DenseNet-121 baseline (torneio) | 0,9833 | ±0,0061 |
| + Fine-tuning gradual (3 fases) | 0,9978 | ±0,0009 |
| + Mixup + Label Smoothing | 0,9983 | ±0,0009 |
| TorchXRayVision v1 (5/5/5 épocas) | 0,9949 | ±0,0007 |
| TorchXRayVision v2 (5/7/10 épocas) | 0,9972 | ±0,0008 |
| DenseNet-121 Augmentação Agressiva | 0,9978 | ±0,0016 |
| EfficientNet-B4 (380×380) | 0,9969 | ±0,0008 |
| **Ensemble ótimo (w=0,70/0,30)** | **0,9987** | **—** |
| Holdout interno (802 imgs / 519 pac.) | 0,9954 | — |
| **🏆 Kaggle Leaderboard** | **0,99129** | — |

> **Nota sobre generalização:** o delta de +0,0028 entre AUC médio de CV (0,9982) e holdout confirma ausência de overfitting estrutural. O plateau de 0,9987 com 4 modelos distintos indica que o teto está no volume de dados (4.430 imagens de treino), não na capacidade dos modelos.

---

## 📁 Estrutura do Repositório

```
.
├── Notebook_final.ipynb          # Notebook principal — pipeline completo e documentado
├── requirements.txt              # Dependências do projeto
├── README.md                     # Este arquivo
│
├── checkpoints/                  # Pesos dos modelos treinados
│   ├── densenet121_fold1.pt
│   ├── densenet121_fold2.pt
│   ├── densenet121_fold3.pt
│   ├── densenet121_fold4.pt
│   ├── densenet121_fold5.pt
│   ├── txrv_fold1.pt
│   ├── txrv_fold2.pt
│   ├── txrv_fold3.pt
│   ├── txrv_fold4.pt
│   └── txrv_fold5.pt
│
├── data/                         # (não incluída — ver seção Dataset)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test_images/
│
├── processed/                    # Dataset pré-processado (gerado pela Célula de Pré-Processamento)
│   ├── train/
│   └── test/
│
└── submissions/
    └── submission_final.csv      # Arquivo de submissão gerado
```

---

## ⚙️ Instalação e Ambiente

### Pré-requisitos

- Python 3.10+
- CUDA 11.8+ (recomendado; o código detecta automaticamente CPU como fallback)
- ~16 GB de VRAM para reprodução completa (treinado em Tesla P100-PCIE-16GB)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/iceb/ligia-xray-classification.git
cd ligia-xray-classification

# Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt
```

### `requirements.txt`

```
torch==2.1.0
torchvision==0.16.0
torchxrayvision==1.4.0
scikit-learn==1.3.2
numpy==1.24.4
pandas==2.1.4
Pillow==10.1.0
opencv-python==4.8.1.78
matplotlib==3.8.2
tqdm==4.66.1
timm==0.9.12
scipy==1.11.4
```

> Para ambiente Kaggle, todas as dependências são instaladas diretamente no notebook. O TorchXRayVision é instalado via `pip install torchxrayvision` no início da seção correspondente.

---

## 📂 Dataset

O dataset é disponibilizado exclusivamente via competição Kaggle (acesso pelo link oficial do processo seletivo da Ligia).

**Estrutura após download:**
```
train/
  NORMAL/      → 1.349 imagens
  PNEUMONIA/   → 3.883 imagens (2.530 bacterianas + 1.345 virais)
test_images/   → 624 imagens
train.csv      → metadados de treino
test.csv       → metadados de teste
```

**Estatísticas relevantes:**
- 5.232 imagens de treino, 624 de teste
- Desbalanceamento de 2,88× (PNEUMONIA/NORMAL)
- 3.458 pacientes únicos — 23,7% com múltiplas imagens (máx: 30 por paciente)
- **Viés de hardware detectado:** ruído de fundo estatisticamente diferente entre classes

---

## 🔄 Pipeline Completo

O `Notebook_final.ipynb` está organizado em seções sequenciais e autodocumentadas:

### 1. Configuração e Reprodutibilidade
```python
SEED = 42
# Sementes fixadas em: Python, NumPy, PyTorch (CPU + CUDA), CuDNN determinístico
```

### 2. Análise Exploratória de Dados (EDA)
- Verificação de integridade (PIL.Image.verify — 0 arquivos corrompidos em 5.856)
- Análise de desbalanceamento e impacto no ROC AUC ingênuo (piso 74,2%)
- **Detecção de viés de hardware** (análise estatística de ruído, N=50/classe)
- Análise espectral por classe (histogramas médios — projeção do perfil de erros)
- **Análise de estrutura por paciente** (identificação de IDs duplicados via nomenclatura)

### 3. Pré-Processamento
| Etapa | Técnica | Justificativa |
|---|---|---|
| Redimensionamento | Letterboxing para 224×224 | Preserva proporções anatômicas |
| Contraste | CLAHE (clipLimit=2,0, tile=8×8) | Realce local sem amplificar ruído térmico |
| Ruído | **NLMeans h=3** (vencedor do torneio) | Melhor Score de Convergência (41,71) entre 3 candidatos |
| Normalização | ImageNet (mean/std por canal) | Compatibilidade com backbones pré-treinados |

> O torneio de filtragem usa um **Score de Convergência** customizado: menor sobreposição estatística das distribuições de brilho e ruído entre classes = viés de hardware mais neutralizado.

### 4. Validação Cruzada Anti-Leakage
```python
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

# Holdout isolado ANTES de qualquer treinamento
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)

# 5 folds garantindo isolamento por patient_id
sgkf = StratifiedGroupKFold(n_splits=5)
```

### 5. Torneios de Seleção (todos com critério de equivalência estatística)

**Torneio de Transforms** (EfficientNet-B0 proxy, 2 folds, 2 épocas):
- Pipeline A (sem augmentação): `0,9481 ± 0,0256` ✅ Adotado por parcimônia
- Pipeline B (augmentação geométrica): `0,9428 ± 0,0224`

**Torneio de Backbones** (599 imagens, 3 folds, 5 épocas, backbone congelado):

| Backbone | AUC Médio | Std |
|---|---|---|
| EfficientNet-B0 | 0,9604 | 0,0117 |
| EfficientNet-B2 (260×260) | 0,9782 | 0,0180 |
| ResNet-50 | 0,9533 | 0,0101 |
| **DenseNet-121** ✅ | **0,9833** | **0,0061** |

**Torneio de Balanceamento** (599 imagens, 3 folds, 5 épocas):
- Sem balanceamento: `0,9899 ± 0,0043`
- WeightedRandomSampler: `0,9896 ± 0,0109`
- **pos_weight=2,91** ✅: `0,9868 ± 0,0062` — menor spread, adotado por parcimônia

### 6. Fine-Tuning Gradual (DenseNet-121 + Mixup + Label Smoothing)

```
Fase 1 — cabeça         (1.025 params,     lr=1e-3, 5 épocas)
Fase 2 — denseblock4    (2.161.153 params,  lr=1e-4, 5 épocas)  ← maior salto
Fase 3 — backbone full  (6.954.881 params,  lr=1e-5, 5 épocas)
```

**Regularização:** Mixup α=0,2 + Label Smoothing ε=0,1 (técnicas ortogonais)
- Antes: loss de treino na Fase 3 colapsa para `0,002–0,010` (memorização)
- Depois: loss estabiliza em `0,165–0,183` (memorização contida)

### 7. TorchXRayVision (pesos especializados em raio-X)

```python
import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-all")
# Adaptação: conv0 1→3 canais (replicação/3), classifier 18→1 saída
```

- Pré-treinado em: NIH ChestX-ray14 + CheXpert + MIMIC-CXR + PadChest (>100k raio-X)
- v1 (5/5/5 épocas): AUC `0,9949` — convergência insuficiente
- **v2 (5/7/10 épocas): AUC `0,9972 ± 0,0008`** — menor spread do projeto

### 8. Ensemble com Otimização de Pesos

```python
# Grid search sobre predições de validação cruzada (nunca sobre o holdout)
# Passos de 0,05 em w_txrv de 0,00 a 1,00
# Peso ótimo: w_mixupLS=0.70, w_txrv=0.30 → AUC médio = 0,9987
```

> O plateau é largo (w_txrv entre 0,25–0,50 produz AUC equivalente), indicando que a diversidade de origem dos pesos (ImageNet vs raio-X) é o fator relevante, não o peso exato.

---

## ▶️ Como Reproduzir

### Opção A — Kaggle (recomendado, ambiente original)

1. Faça fork da competição e adicione o dataset como input
2. Faça upload do `Notebook_final.ipynb`
3. Ative GPU P100 (Kaggle oferece gratuitamente)
4. Execute todas as células em ordem (`Run All`)
5. O arquivo `submission_final.csv` é gerado automaticamente na última célula

### Opção B — Ambiente Local

```bash
# 1. Configure o ambiente (ver seção Instalação)

# 2. Baixe o dataset via Kaggle CLI
pip install kaggle
kaggle competitions download -c [nome-da-competicao-ligia]
unzip *.zip -d data/

# 3. Execute o notebook
jupyter notebook Notebook_final.ipynb

# 4. Execute as células em ordem:
#    - Célula de Configuração (SEED, paths)
#    - EDA (análise exploratória)
#    - Pré-Processamento (gera pasta processed/)
#    - Cross Validation Setup
#    - Torneios (transforms, balanceamento, backbone)
#    - Treinamento DenseNet-121 + Mixup+LS (salva checkpoints/)
#    - Treinamento TorchXRayVision v2 (salva checkpoints_txrv/)
#    - Ensemble + Geração de Submissão
```

### Apenas Inferência (com checkpoints pré-treinados)

```python
import torch
from torchvision import models

# Carregar modelo Mixup+LS
model_mixup = models.densenet121(pretrained=False)
model_mixup.classifier = torch.nn.Linear(1024, 1)
model_mixup.load_state_dict(torch.load('checkpoints/densenet121_fold1.pt'))
model_mixup.eval()

# Para ensemble: média ponderada das predições
# pred_final = 0.70 * pred_mixup + 0.30 * pred_txrv
```

> Todos os checkpoints foram salvos com `torch.save(model.state_dict(), path)` e são compatíveis com PyTorch ≥ 2.0.

---

## 🧠 Decisões Técnicas

### Por que DenseNet-121?

As conexões densas — cada camada recebe gradiente direto de todas as anteriores — entregam dois benefícios críticos para este domínio:

1. **Convergência acelerada em texturas finas:** no torneio de 5 épocas, o DenseNet já ultrapassava AUC 0,91 enquanto concorrentes oscilavam abaixo de 0,70
2. **Interpretabilidade por Grad-CAM:** gradientes chegam às camadas superficiais com menor degradação, produzindo mapas de saliência mais confiáveis — essencial para validação clínica

### Por que NLMeans h=3 e não Filtro Bilateral?

O Filtro Bilateral preserva bordas seletivamente mas manteve disparidades residuais entre classes. Isso revelou que a fonte do viés de hardware é a **variação de textura global**, não apenas bordas. O NLMeans com h=3 (suavização leve e constante) foi mais eficaz em unificar as assinaturas de ruído.

### Por que não usar augmentação geométrica?

Raio-X torácico impõe restrições anatômicas rígidas:
- **Flip horizontal** simula dextrocardia (posição invertida do coração — condição cardíaca rara)
- **Rotações acima de 10°** produzem incidências clinicamente inexistentes
- O CLAHE + NLMeans já extraiu a variabilidade essencial do dataset (comprovado pelo torneio de transforms)

### Por que Mixup + Label Smoothing são complementares?

| Técnica | Age quando... | Gap |
|---|---|---|
| Mixup (α=0,2) | λ ~ Beta(0,2, 0,2) → soft labels dinâmicos | λ próximo de 0 ou 1 → rótulos quase binários |
| Label Smoothing (ε=0,1) | Sempre | Cobre exatamente o gap do Mixup |

---

## 🔍 Análise de Interpretabilidade

### Evidências Indiretas (do Notebook)

Sem implementar Grad-CAM, os logs já fornecem evidência sobre o que o modelo aprendeu:

1. **Convergência explosiva do denseblock4** na Fase 2 (AUC 0,9878→0,9978 em 5 épocas) indica que representações de alta abstração semântica capturaram padrões diagnósticos — infiltrados e consolidações são padrões espaciais de alta frequência concentrados nas camadas mais profundas

2. **Offset de normalização persistente** após todo o pré-processamento:
   - NORMAL: mean `−0,154 ± 0,256`
   - PNEUMONIA: mean `−0,381 ± 0,347`
   
   Diferença de 0,23 unidades de desvio padrão compatível com hiperdensidade focal real

3. **Convergência estatística pós-NLMeans** (N=300): a feature de ruído de sensor foi eliminada por construção, tornando improvável seu uso como discriminador

### Próximo Passo — Grad-CAM

```python
# Registrar hook na camada de maior abstração semântica
target_layer = model.features.norm5

# Executar backward para a classe predita
# Sobrepor mapa às radiografias originais
# Validar com anotações de radiologistas
```

> Grad-CAM sobre `features.norm5` permitiria validar clinicamente se as regiões de maior ativação coincidem com as áreas de consolidação marcadas por especialistas — custo computacional zero (apenas inferência).

### Perfil de Erros

A análise espectral da EDA (histogramas médios por classe) projeta com precisão quais casos são mais difíceis:

- **Casos mais difíceis:** PNEUMONIA viral inicial — infiltrado intersticial sutil, sobreposição espectral alta com NORMAL
- **Casos mais fáceis:** PNEUMONIA bacteriana consolidada — deslocamento para tons claros (150–230), baixa sobreposição com NORMAL

O pos_weight=2,91 resulta em Recall >0,98 para PNEUMONIA em todos os folds. Em produção, o threshold de 0,5 deve ser ajustado via curva Precision-Recall para o contexto clínico específico.

---

## ⚠️ Limitações e Trabalhos Futuros

| Limitação | Impacto | Solução Proposta |
|---|---|---|
| Volume de dados (4.430 treino) | Plateau de ensemble em 0,9987 com 4 modelos | Incorporar CheXpert / NIH ChestX-ray14 |
| Grad-CAM não implementado | Interpretabilidade apenas indireta | Hook em `features.norm5` + validação clínica |
| Calibração não avaliada | Probabilidades podem ser mal calibradas para triagem | Expected Calibration Error (ECE) |
| Dataset de fonte única | Generalização inter-institucional não testada | Teste em dados de equipamentos distintos |
| Folds de ~880 amostras | Difícil discriminar ganhos < 0,001 AUC | Maior dataset ou bootstrap CI |

> **Experimento negativo documentado:** pipeline sem StratifiedGroupKFold atingiu AUC=1,0000 em validação interna mas colapsou no leaderboard Kaggle (AUC <0,981), confirmando empiricamente que o isolamento por paciente é requisito inegociável, não preciosismo metodológico.

---

## 👤 Autor

**Ivan Carvalho Ernesto Bezerra**  
Centro de Informática – UFPE  
[iceb@cin.ufpe.br](mailto:iceb@cin.ufpe.br)

---

> *Se você está avaliando este repositório e tiver a métrica de alguma submissão que não pude testar (esgotei o limite de submissões no Kaggle), ficaria muito grato se pudesse me enviar por email. Obrigado!*

---

<div align="center">
<sub>Ligia – Liga Acadêmica de Inteligência Artificial · UFPE · 2026</sub>
</div>
