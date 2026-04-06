"""
merge_lora.py — Merge des poids LoRA dans le modèle de base

Charge le modèle de base et les poids LoRA, les fusionne,
et sauvegarde le modèle complet fusionné sur disque.

Le modèle résultant est un modèle standard (sans PEFT) utilisable
directement avec from_pretrained, llama.cpp, vllm, ou n'importe quel
outil d'inférence sans dépendance à peft.

Usage :
    python merge_lora.py \\
        --base Qwen/Qwen2.5-7B-Instruct \\
        --adapter output/qwen2.5-7b-lora \\
        --output output/qwen2.5-7b-merged

Notes :
    - Le merge se fait en fp16 par défaut.
    - Le modèle de base doit être accessible (HuggingFace Hub ou local).
    - L'adapteur doit être un dossier contenant les fichiers PEFT
      (adapter_config.json, adapter_model.safetensors).
    - La VRAM nécessaire = modèle de base en fp16 (~14 Go pour 7B).
      Si la VRAM est insuffisante, utiliser --device cpu (lent mais fonctionne).
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge les poids LoRA dans le modèle de base et sauvegarde le résultat."
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Identifiant HuggingFace ou chemin local du modèle de base (sans LoRA).",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        type=Path,
        help="Chemin vers le dossier contenant les poids LoRA (adapter_config.json, etc.).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Dossier de sortie pour le modèle mergé.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help=(
            "Device pour le chargement du modèle. "
            "'auto' utilise le GPU si disponible (défaut). "
            "'cpu' est plus lent mais ne nécessite pas de VRAM."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Précision du modèle mergé (défaut: bf16).",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convertit une chaîne de type en torch.dtype."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_str]


def main() -> None:
    args = parse_args()

    # Vérification du dossier adapteur
    if not args.adapter.exists():
        raise FileNotFoundError(
            f"Le dossier d'adapteur '{args.adapter}' n'existe pas. "
            "Vérifier que l'entraînement s'est bien terminé."
        )

    adapter_config = args.adapter / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Fichier adapter_config.json introuvable dans '{args.adapter}'. "
            "Ce dossier ne semble pas être un adapteur PEFT valide."
        )

    args.output.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(args.dtype)

    # Chargement du modèle de base (sans quantification — on veut les poids en pleine précision)
    print(f"Chargement du modèle de base : {args.base}")
    print(f"  Device : {args.device} | Dtype : {args.dtype}")

    if args.device == "cpu":
        # Sur CPU, charger directement en fp32 ou fp16 selon la RAM disponible
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            device_map=args.device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    # Chargement du tokenizer
    # On charge le tokenizer depuis l'adapteur en priorité (peut avoir été modifié)
    # avec fallback sur le modèle de base si absent
    tokenizer_path = args.adapter if (args.adapter / "tokenizer_config.json").exists() else args.base
    print(f"Chargement du tokenizer : {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
    )

    # Chargement de l'adapteur LoRA par-dessus le modèle de base
    print(f"Chargement de l'adapteur LoRA : {args.adapter}")
    model = PeftModel.from_pretrained(
        model,
        str(args.adapter),
        torch_dtype=dtype,
    )

    # Merge et déchargement des poids LoRA
    # merge_and_unload() fusionne les matrices A*B dans les poids W originaux
    # et retourne un modèle standard sans couches PEFT
    print("Fusion des poids LoRA dans le modèle de base...")
    model = model.merge_and_unload()

    # Le modèle résultant est maintenant un AutoModelForCausalLM standard
    # sans aucune dépendance à PEFT

    # Sauvegarde du modèle mergé
    print(f"Sauvegarde du modèle mergé -> {args.output}")
    model.save_pretrained(
        str(args.output),
        safe_serialization=True,   # Sauvegarder en format safetensors (plus sûr que pickle)
        max_shard_size="4GB",      # Découper en shards de 4 Go pour faciliter le partage
    )
    tokenizer.save_pretrained(str(args.output))

    print("\nMerge terminé.")
    print(f"Modèle complet disponible dans : {args.output}")
    print(
        "\nPour tester le modèle mergé :\n"
        "  from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f'  model = AutoModelForCausalLM.from_pretrained("{args.output}", device_map="auto")\n'
        f'  tokenizer = AutoTokenizer.from_pretrained("{args.output}")'
    )


if __name__ == "__main__":
    main()
