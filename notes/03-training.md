# 03 — Configuration d'entraînement LoRA

## Architecture de l'entraînement

Le pipeline repose sur trois librairies HuggingFace :
- `transformers` : chargement du modèle, tokenizer, `apply_chat_template`
- `peft` : application de LoRA sur le modèle (`LoraConfig`, `get_peft_model`)
- `trl` : `SFTTrainer`, qui gère la boucle d'entraînement supervisé

Pour QLoRA, on ajoute `bitsandbytes` pour la quantification NF4.

---

## Paramètres LoRA

### Rank (r)

Le rang des matrices LoRA. Contrôle le nombre de paramètres entraînables.

- `r=8` : léger, adapté aux adaptations de style simples, très peu de VRAM supplémentaire
- `r=16` : bon équilibre général, recommandé pour débuter
- `r=32` : plus de capacité d'adaptation, ralentit légèrement l'entraînement
- `r=64` : rarement nécessaire sauf dataset > 10k exemples avec forte variance

Pour Qwen2.5-7B-Instruct sur 200-500 exemples : `r=16` suffit.

### Alpha (lora_alpha)

Facteur de scaling appliqué aux matrices LoRA lors de la forward pass.
En pratique : `alpha = 2 * r` est un point de départ raisonnable.

- `r=16, alpha=32` : configuration la plus courante
- Augmenter alpha accélère l'apprentissage mais peut déstabiliser l'entraînement

### Dropout

`lora_dropout=0.05` est une valeur conservative.
Sur des petits datasets (< 500 exemples), monter à `0.1` pour réduire l'overfitting.
Sur des datasets > 2000 exemples, `0.0` est acceptable.

### Target modules pour Qwen2.5

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

Cibler uniquement `q_proj` et `v_proj` réduit de 50% les paramètres entraînables.
Acceptable pour des adaptations très légères, mais sous-optimal pour la plupart des cas.

---

## Hyperparamètres d'entraînement

### Learning rate

- `2e-4` est le point de départ standard pour LoRA
- `1e-4` si l'entraînement est instable (loss qui monte ou oscille)
- `5e-4` pour accélérer sur un dataset propre et homogène

Toujours utiliser un scheduler. `cosine` avec warmup est le plus courant.

```python
learning_rate = 2e-4
lr_scheduler_type = "cosine"
warmup_ratio = 0.03  # 3% des steps en warmup
```

### Batch size et gradient accumulation

Le batch size effectif = `per_device_train_batch_size * gradient_accumulation_steps`.

Recommandation : viser un batch size effectif de 16-32.

| VRAM disponible | batch_size | grad_accum | Batch effectif |
|---|---|---|---|
| 8-12 Go (QLoRA) | 1 | 16 | 16 |
| 16 Go (QLoRA) | 2 | 8 | 16 |
| 24 Go (LoRA fp16) | 4 | 4 | 16 |
| 24 Go (LoRA fp16) | 8 | 2 | 16 |

Un batch effectif trop petit (< 8) rend les gradients bruités et l'entraînement instable.
Un batch effectif trop grand (> 64) ralentit la convergence.

### Nombre d'epochs

- 1-3 epochs sur des petits datasets (< 500 exemples) : suffisant, au-delà l'overfitting est probable
- 1-2 epochs sur des datasets moyens (500-5000 exemples)
- 1 epoch sur des datasets larges (> 10k exemples)

Surveiller la validation loss pour stopper avant l'overfitting.
Activer `load_best_model_at_end=True` dans `TrainingArguments`.

### Longueur de séquence max

`max_seq_length=2048` est le défaut raisonnable pour Qwen2.5-7B.
Augmenter à `4096` si les exemples sont longs (code, documents).
Dépasse rarement le besoin de monter au-delà de `4096` pour du fine-tuning supervisé.

---

## Gradient checkpointing

Active par défaut dans le script. Réduit la VRAM utilisée d'environ 30-40%
en recalculant les activations intermédiaires pendant le backward pass
au lieu de les stocker.

Contrepartie : ralentit l'entraînement de 20-30%.
Si la VRAM n'est pas un facteur limitant, désactiver pour aller plus vite :
`gradient_checkpointing=False`.

---

## Mixed precision

- `bf16=True` sur les GPU Ampere+ (RTX 30xx, RTX 40xx, A100) : recommandé
- `fp16=True` sur les GPU Turing (RTX 20xx) ou Volta (V100) : utiliser à la place
- Les deux en même temps : erreur. Un seul.

Pour QLoRA avec bitsandbytes, utiliser `bf16=True` — la quantification NF4
est optimisée pour bf16.

---

## Sauvegarde et checkpoints

```python
save_strategy = "epoch"         # Sauvegarder à chaque fin d'epoch
save_total_limit = 2            # Garder les 2 derniers checkpoints
load_best_model_at_end = True   # Charger le meilleur checkpoint à la fin
metric_for_best_model = "eval_loss"
```

Les checkpoints contiennent seulement les poids LoRA (quelques centaines de Mo),
pas le modèle complet. Le merge avec le modèle de base se fait après coup
avec `scripts/merge_lora.py`.

---

## Profil VRAM complet — Qwen2.5-7B-Instruct

| Configuration | VRAM modèle | VRAM gradients | VRAM total estimé |
|---|---|---|---|
| LoRA fp16, batch=1 | ~14 Go | ~2 Go | ~16-18 Go |
| LoRA fp16, batch=4 | ~14 Go | ~6 Go | ~20-22 Go |
| QLoRA 4-bit, batch=1 | ~4.5 Go | ~2 Go | ~7-9 Go |
| QLoRA 4-bit, batch=2 | ~4.5 Go | ~3 Go | ~8-10 Go |

Ces valeurs varient selon la longueur des séquences et l'activation du gradient checkpointing.
Toujours démarrer avec batch=1 et monter progressivement.
