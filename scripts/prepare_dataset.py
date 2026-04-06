"""
prepare_dataset.py — Conversion d'un dataset brut en format Alpaca

Prend un fichier JSONL en entrée (un objet JSON par ligne)
et produit un fichier JSON au format Alpaca (instruction/input/output).

Applique les filtrages basiques :
- Suppression des doublons sur le champ instruction
- Filtrage des réponses trop courtes (< min_output_tokens tokens)
- Filtrage des exemples dont l'instruction ou la réponse est vide
- Rapport des statistiques de nettoyage en fin d'exécution

Usage :
    python prepare_dataset.py \\
        --input data/raw/exemples.jsonl \\
        --output data/processed/train.json \\
        --instruction-field "question" \\
        --output-field "answer" \\
        --min-output-tokens 10

Champs par défaut attendus dans le JSONL :
    instruction, input (optionnel), output

Pour mapper d'autres noms de champs, utiliser les options --*-field.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convertit un dataset JSONL brut en format Alpaca."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Chemin vers le fichier JSONL d'entrée.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Chemin vers le fichier JSON de sortie.",
    )
    parser.add_argument(
        "--instruction-field",
        default="instruction",
        help="Nom du champ instruction dans le JSONL source (défaut: instruction).",
    )
    parser.add_argument(
        "--input-field",
        default="input",
        help="Nom du champ input dans le JSONL source (défaut: input). Optionnel.",
    )
    parser.add_argument(
        "--output-field",
        default="output",
        help="Nom du champ output dans le JSONL source (défaut: output).",
    )
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=10,
        help="Nombre minimal de tokens (approx. mots) dans la réponse (défaut: 10).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Nombre maximal de tokens (approx. mots) dans la réponse (défaut: 2048).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Désactiver la déduplication sur le champ instruction.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        help="Chemin optionnel pour exporter un split d'évaluation.",
    )
    parser.add_argument(
        "--eval-size",
        type=float,
        default=0.1,
        help="Part du dataset réservée à l'évaluation si --eval-output est utilisé (défaut: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed utilisée pour le split train/eval (défaut: 42).",
    )
    return parser.parse_args()


def approximate_token_count(text: str) -> int:
    """
    Approximation grossière du nombre de tokens.
    1 token ~ 0.75 mot en anglais, ~0.6 mot en français.
    On divise par 0.65 comme compromis.
    Pour une estimation précise, utiliser le tokenizer du modèle.
    """
    words = text.split()
    return int(len(words) / 0.65)


def load_jsonl(path: Path) -> list[dict]:
    """
    Charge un fichier JSONL.
    Chaque ligne doit être un objet JSON valide.
    Les lignes vides sont ignorées.
    """
    records = []
    errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"  [AVERTISSEMENT] Ligne {line_number} ignorée (JSON invalide) : {e}",
                    file=sys.stderr,
                )
                errors += 1

    if errors > 0:
        print(
            f"  {errors} ligne(s) ignorée(s) à cause d'erreurs JSON.",
            file=sys.stderr,
        )

    return records


def convert_to_alpaca(
    record: dict,
    instruction_field: str,
    input_field: str,
    output_field: str,
) -> dict | None:
    """
    Convertit un enregistrement brut en format Alpaca.
    Retourne None si les champs obligatoires sont manquants ou vides.
    """
    instruction = str(record.get(instruction_field, "")).strip()
    output = str(record.get(output_field, "")).strip()
    input_text = str(record.get(input_field, "")).strip()

    # Les deux champs obligatoires doivent être non vides
    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input": input_text,  # Peut être vide — c'est valide en Alpaca
        "output": output,
    }


def deduplicate(
    records: list[dict],
    key: str = "instruction",
) -> tuple[list[dict], int]:
    """
    Supprime les doublons en se basant sur un champ clé.
    Conserve la première occurrence.
    Retourne (records_dédupliqués, nombre_de_doublons_supprimés).
    """
    seen: set[str] = set()
    unique: list[dict] = []
    duplicates = 0

    for record in records:
        value = record.get(key, "").strip().lower()
        if value in seen:
            duplicates += 1
            continue
        seen.add(value)
        unique.append(record)

    return unique, duplicates


def filter_by_length(
    records: list[dict],
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[dict], int]:
    """
    Filtre les exemples dont la réponse est trop courte ou trop longue.
    Retourne (records_filtrés, nombre_d_exemples_supprimés).
    """
    filtered = []
    removed = 0

    for record in records:
        output = record.get("output", "")
        token_count = approximate_token_count(output)

        if token_count < min_tokens or token_count > max_tokens:
            removed += 1
            continue

        filtered.append(record)

    return filtered, removed


def split_records(
    records: list[dict],
    eval_size: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Sépare le dataset en deux sous-ensembles reproductibles.
    Retourne (train_records, eval_records).
    """
    if not 0 < eval_size < 1:
        raise ValueError("--eval-size doit être strictement compris entre 0 et 1.")

    if len(records) < 2:
        raise ValueError(
            "Au moins 2 enregistrements valides sont nécessaires pour créer un split train/eval."
        )

    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    eval_count = max(1, int(len(shuffled) * eval_size))
    eval_records = shuffled[:eval_count]
    train_records = shuffled[eval_count:]

    if not train_records:
        raise ValueError("Le split demandé ne laisse aucun exemple dans l'ensemble train.")

    return train_records, eval_records


def main() -> None:
    args = parse_args()

    # Vérification de l'entrée
    if not args.input.exists():
        print(f"Erreur : le fichier d'entrée '{args.input}' n'existe pas.", file=sys.stderr)
        sys.exit(1)

    # Créer le dossier de sortie si nécessaire
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Chargement de : {args.input}")
    raw_records = load_jsonl(args.input)
    print(f"  {len(raw_records)} enregistrements chargés.")

    # Conversion au format Alpaca
    converted = []
    skipped_empty = 0

    for record in raw_records:
        alpaca_record = convert_to_alpaca(
            record,
            instruction_field=args.instruction_field,
            input_field=args.input_field,
            output_field=args.output_field,
        )
        if alpaca_record is None:
            skipped_empty += 1
        else:
            converted.append(alpaca_record)

    print(f"  {skipped_empty} enregistrements ignorés (instruction ou output vide).")
    print(f"  {len(converted)} enregistrements convertis au format Alpaca.")

    # Déduplication
    if not args.no_dedup:
        converted, duplicates = deduplicate(converted, key="instruction")
        print(f"  {duplicates} doublon(s) supprimé(s) sur le champ 'instruction'.")
    else:
        print("  Déduplication désactivée.")

    # Filtrage par longueur
    converted, removed_length = filter_by_length(
        converted,
        min_tokens=args.min_output_tokens,
        max_tokens=args.max_output_tokens,
    )
    print(
        f"  {removed_length} enregistrements supprimés (output < {args.min_output_tokens} "
        f"ou > {args.max_output_tokens} tokens approx.)."
    )

    if not converted:
        print(
            "Erreur : aucun enregistrement valide après nettoyage. "
            "Vérifier les noms de champs et les seuils de filtrage.",
            file=sys.stderr,
        )
        sys.exit(1)

    final_records = converted
    eval_records: list[dict] = []

    if args.eval_output is not None:
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        final_records, eval_records = split_records(
            converted,
            eval_size=args.eval_size,
            seed=args.seed,
        )
        with open(args.eval_output, "w", encoding="utf-8") as f:
            json.dump(eval_records, f, ensure_ascii=False, indent=2)
        print(
            f"Split train/eval créé : train={len(final_records)} | eval={len(eval_records)} "
            f"(seed={args.seed})"
        )
        print(f"Jeu d'évaluation écrit dans : {args.eval_output}")

    # Sauvegarde
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)

    print(f"\nDataset final : {len(final_records)} exemples -> {args.output}")

    # Distribution des longueurs (informatif)
    lengths = [approximate_token_count(r["output"]) for r in final_records]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)
    print(f"Longueur output (tokens approx.) : min={min_length}, max={max_length}, moy={avg_length:.0f}")


if __name__ == "__main__":
    main()
