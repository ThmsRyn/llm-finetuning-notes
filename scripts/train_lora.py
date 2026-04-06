"""
train_lora.py — Entraînement LoRA / QLoRA avec SFTTrainer

Entraîne un adapteur LoRA sur Qwen2.5-7B-Instruct (ou tout autre modèle
compatible) à partir d'un dataset au format Alpaca.

Supporte LoRA en fp16/bf16 et QLoRA en 4-bit NF4 via bitsandbytes.

Usage :
    # LoRA standard (GPU >= 20 Go VRAM)
    python train_lora.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset data/processed/train.json \\
        --output output/qwen2.5-7b-lora

    # QLoRA 4-bit (GPU >= 8 Go VRAM)
    python train_lora.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset data/processed/train.json \\
        --output output/qwen2.5-7b-lora \\
        --use-qlora

Dépendances : transformers, peft, trl, datasets, torch, bitsandbytes (pour QLoRA)
"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement LoRA/QLoRA sur un dataset Alpaca."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Identifiant HuggingFace ou chemin local du modèle de base.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Chemin vers le dataset JSON (format Alpaca : liste d'objets instruction/input/output).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Dossier de sortie pour les poids LoRA et les checkpoints.",
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="Activer QLoRA (quantification 4-bit NF4). Recommandé si VRAM < 16 Go.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="Rang LoRA (défaut: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="Alpha LoRA — facteur de scaling (défaut: 32 = 2x rank).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout sur les couches LoRA (défaut: 0.05).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Longueur maximale des séquences (défaut: 2048).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size par GPU (défaut: 1). Augmenter si VRAM le permet.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (défaut: 16). Batch effectif = batch_size * grad_accum.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Nombre d'epochs (défaut: 3).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (défaut: 2e-4).",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction du dataset réservée à l'évaluation (défaut: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed aléatoire pour la reproductibilité.",
    )
    return parser.parse_args()


def load_alpaca_dataset(path: Path) -> Dataset:
    """
    Charge un dataset au format Alpaca depuis un fichier JSON.
    Retourne un objet HuggingFace Dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            f"Le fichier {path} doit contenir une liste d'objets JSON (format Alpaca)."
        )

    return Dataset.from_list(records)


def format_alpaca_prompt(example: dict, tokenizer) -> dict:
    """
    Formate un exemple Alpaca en appliquant le chat template de Qwen.

    Le format ChatML de Qwen est appliqué automatiquement via apply_chat_template.
    Cela évite de construire manuellement les tokens spéciaux <|im_start|> / <|im_end|>.
    """
    # Construction du message utilisateur
    if example.get("input", "").strip():
        user_content = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_content = example["instruction"]

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]

    # apply_chat_template ajoute les tokens spéciaux Qwen (ChatML)
    # tokenize=False pour obtenir le texte brut — SFTTrainer tokenise ensuite
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": formatted}


def build_qlora_config() -> BitsAndBytesConfig:
    """
    Configuration QLoRA : quantification NF4 en double quantification.
    Double quantification (bnb_4bit_use_double_quant=True) réduit encore
    la VRAM de ~0.4 bits par paramètre supplémentaire.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — meilleur que int4 pour les LLMs
        bnb_4bit_use_double_quant=True,       # Double quantification pour réduire encore la VRAM
        bnb_4bit_compute_dtype=torch.bfloat16, # Calculs en bf16 pendant le forward/backward
    )


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """
    Configuration LoRA pour Qwen2.5.

    target_modules couvre toutes les projections d'attention et les projections MLP.
    Pour un fine-tuning plus léger, réduire à ["q_proj", "v_proj"].
    """
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",                    # Ne pas entraîner les biais — recommandé pour LoRA
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def main() -> None:
    args = parse_args()

    # Vérification de l'environnement
    if not torch.cuda.is_available():
        print(
            "AVERTISSEMENT : aucun GPU CUDA détecté. L'entraînement sur CPU est "
            "extrêmement lent et déconseillé."
        )
    else:
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU : {device_name} ({vram_total:.1f} Go VRAM)")

    args.output.mkdir(parents=True, exist_ok=True)

    # Chargement du tokenizer
    print(f"\nChargement du tokenizer : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,  # Requis pour Qwen
        padding_side="right",    # Padding à droite pour les modèles causaux
    )
    # Qwen2.5 utilise <|endoftext|> comme pad token par défaut
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Chargement du modèle
    print(f"Chargement du modèle : {args.model}")
    if args.use_qlora:
        print("  Mode QLoRA 4-bit activé.")
        bnb_config = build_qlora_config()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",           # Répartit automatiquement sur les GPU disponibles
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        print("  Mode LoRA fp16/bf16.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Activer le gradient checkpointing pour réduire la VRAM
    # use_reentrant=False est recommandé avec PEFT pour éviter des bugs de gradient
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Application de la configuration LoRA
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)

    # Affichage des paramètres entraînables
    model.print_trainable_parameters()

    # Chargement et préparation du dataset
    print(f"\nChargement du dataset : {args.dataset}")
    dataset = load_alpaca_dataset(args.dataset)
    print(f"  {len(dataset)} exemples chargés.")

    # Formatage avec le chat template Qwen
    dataset = dataset.map(
        lambda example: format_alpaca_prompt(example, tokenizer),
        remove_columns=dataset.column_names,  # Garder seulement le champ "text"
        desc="Application du chat template",
    )

    # Split train/eval
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train : {len(train_dataset)} exemples | Eval : {len(eval_dataset)} exemples")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  Train : {len(train_dataset)} exemples (pas de split eval)")

    # Calcul du nombre de steps pour information
    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps par epoch : {steps_per_epoch} | Total : {total_steps}")

    # Configuration de l'entraînement
    # SFTConfig hérite de TrainingArguments avec des paramètres supplémentaires pour SFT
    sft_config = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",             # Scheduler cosine avec warmup
        warmup_ratio=0.03,                      # 3% des steps en warmup
        weight_decay=0.01,                      # Régularisation légère
        bf16=True,                              # bf16 pour GPU Ampere+ (RTX 30xx/40xx)
        # fp16=True,                            # Décommenter pour GPU Turing (RTX 20xx)
        logging_steps=10,                       # Log la loss tous les 10 steps
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,                     # Garder les 2 derniers checkpoints
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        dataloader_num_workers=0,              # 0 sur Windows, peut monter sur Linux
        report_to="none",                      # Désactiver wandb/tensorboard par défaut
        # Paramètres spécifiques SFT
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",             # Champ contenant le texte formaté
        packing=False,                          # Packing désactivé — plus simple pour débuter
    )

    # Initialisation du SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Lancement de l'entraînement
    print(f"\nDémarrage de l'entraînement...")
    print(f"  Batch effectif : {args.batch_size * args.grad_accum}")
    print(f"  Learning rate : {args.learning_rate}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Output : {args.output}\n")

    trainer.train()

    # Sauvegarde des poids LoRA finaux
    print(f"\nSauvegarde des poids LoRA -> {args.output}")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    print("\nEntraînement terminé.")
    print(f"Poids LoRA sauvegardés dans : {args.output}")
    print(
        "Pour merger les poids LoRA avec le modèle de base, "
        "utiliser scripts/merge_lora.py"
    )


if __name__ == "__main__":
    main()
