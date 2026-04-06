# llm-finetuning-notes

Notes et scripts autour du fine-tuning de modèles Qwen avec LoRA et QLoRA.
Basé sur de l'expérience réelle : entraînements sur GPU, datasets maison, merges de poids.

Ce repo n'est pas un tutoriel générique copié-collé de la documentation HuggingFace.
C'est ce qui a fonctionné, ce qui a planté, et pourquoi.

---

## Ce que couvre ce repo

- Comparaison fine-tuning / RAG / prompting — quand choisir quoi
- LoRA vs QLoRA — compromis VRAM vs qualité
- Préparation de datasets : formats Alpaca, ShareGPT, ChatML
- Configuration d'entraînement pour Qwen2.5-7B-Instruct
- Evaluation d'un modèle fine-tuné : perplexité, comparaison manuelle, overfitting
- Scripts fonctionnels pour préparer les données, lancer l'entraînement, merger les poids

---

## Modèle de référence

[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — HuggingFace

Les configs et scripts sont calibrés pour ce modèle.
Les principes s'appliquent à d'autres modèles Qwen (1.5B, 3B, 14B, 72B)
avec les ajustements de VRAM correspondants.

---

## Prérequis matériels

| Config | Ce que ca permet |
|---|---|
| 1x GPU 12 Go VRAM (RTX 3060/4060) | QLoRA 4-bit, batch size 1-2 |
| 1x GPU 24 Go VRAM (RTX 3090/4090) | LoRA fp16, batch size 4-8, ou QLoRA confortable |
| 2x GPU 24 Go | LoRA fp16 avec gradient checkpointing desactivé |
| CPU seul | Possible pour tester les scripts, pas pour entraîner |

Les scripts sont configurés pour une seule GPU.
Pour du multi-GPU, adapter `device_map` et ajouter `accelerate`.

---

## Prérequis logiciels

- Python 3.11+
- CUDA 12.1+ (driver GPU a jour)
- pip install -r requirements.txt

Environnement testé :
- Ubuntu 22.04 / Debian 12
- CUDA 12.2
- Driver NVIDIA 535+

---

## Structure

```
llm-finetuning-notes/
├── notes/
│   ├── 01-concepts.md          # Fine-tuning vs RAG vs prompting, LoRA vs QLoRA
│   ├── 02-dataset-prep.md      # Formats, nettoyage, taille, qualite des exemples
│   ├── 03-training.md          # Config LoRA pour Qwen, hyperparamètres, VRAM
│   └── 04-evaluation.md        # Perplexité, evaluation manuelle, pieges classiques
├── scripts/
│   ├── prepare_dataset.py      # Conversion dataset brut -> format Alpaca
│   ├── train_lora.py           # Entraînement LoRA avec SFTTrainer
│   └── merge_lora.py           # Merge des poids LoRA dans le modèle de base
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Utilisation rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Préparer le dataset
python scripts/prepare_dataset.py \
  --input data/raw/mes_exemples.jsonl \
  --output data/processed/train.json

# 3. Lancer l'entraînement
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/processed/train.json \
  --output output/qwen2.5-7b-lora

# 4. Merger les poids
python scripts/merge_lora.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --adapter output/qwen2.5-7b-lora \
  --output output/qwen2.5-7b-merged
```

---

## Licence

MIT — voir `LICENSE`.
