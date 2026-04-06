# 01 — Concepts de base

## Fine-tuning vs RAG vs prompting

Ces trois approches ne sont pas concurrentes. Elles répondent à des problèmes différents.
Choisir la mauvaise approche coûte du temps, de l'argent GPU, et ne résout pas le vrai problème.

### Prompting

Le prompting (system prompt, few-shot examples dans le contexte) est suffisant quand :
- Le modèle de base sait déjà faire ce que tu veux, il faut juste le cadrer
- Tu veux un ton particulier, un format de réponse précis, un rôle spécifique
- Ton cas d'usage change souvent et tu ne veux pas ré-entraîner

Limites : le contexte a une taille finie (même avec 32k tokens), les instructions se diluent
sur les longues conversations, et le modèle peut "oublier" ses instructions si le contexte est chargé.

### RAG (Retrieval-Augmented Generation)

Le RAG est la bonne réponse quand :
- Tu as une base de connaissance volumineuse et changeante (documentation, tickets, emails)
- Le modèle doit répondre avec des faits précis qu'il ne peut pas avoir mémorisés
- Tu veux de la traçabilité (savoir quels documents ont été utilisés pour répondre)

Le RAG ne change pas le comportement du modèle, il lui fournit du contexte au moment de l'inférence.
Si le modèle est mauvais pour synthétiser, le RAG ne corrige pas ca.

Limites : latence (retrieval + inférence), qualité dépendante de l'index vectoriel,
et inutile si ce que tu veux c'est changer le *style* ou les *capacités* du modèle.

### Fine-tuning

Le fine-tuning est justifié quand :
- Tu veux que le modèle adopte un style d'écriture spécifique de façon systématique
- Tu veux lui apprendre un format de sortie structuré (JSON avec un schema précis, code dans un style particulier)
- Tu travailles sur du domaine très spécifique avec une terminologie que le modèle de base gère mal
- Tu veux un modèle plus petit qui performe aussi bien qu'un grand sur une tâche étroite

Ce que le fine-tuning ne fait pas : il n'injecte pas de connaissances factuelles de façon fiable.
Essayer de faire mémoriser des faits précis au modèle par fine-tuning mène à de l'hallucination
avec plus de confiance. Pour les faits, c'est le RAG.

---

## LoRA — Low-Rank Adaptation

Le fine-tuning classique met à jour tous les poids du modèle.
Pour un modèle 7B, ca représente des milliards de paramètres à stocker et à optimiser.
Impossible sur une GPU grand public.

LoRA contourne ca : au lieu de modifier les matrices de poids existantes,
on ajoute des matrices de faible rang en parallèle. Ces matrices additionnelles
sont petites (quelques millions de paramètres) et c'est elles qu'on entraîne.

Le principe : toute matrice peut être approximée par le produit de deux matrices plus petites.
Si W est une matrice (d x d), on entraîne A (d x r) et B (r x d) où r << d.
Delta_W = A * B. Seuls A et B sont entraînés, W reste gelé.

En pratique pour Qwen2.5-7B avec LoRA rank=16 :
- Modèle complet : ~14 Go en fp16
- Adapteur LoRA : ~100-300 Mo selon le rank et les modules ciblés
- VRAM nécessaire : ~18-20 Go en fp16 avec gradient checkpointing

---

## QLoRA — Quantized LoRA

QLoRA combine deux techniques :
1. Le modèle de base est chargé en 4-bit (NF4 — NormalFloat4) via bitsandbytes
2. L'entraînement LoRA se fait par-dessus ce modèle quantifié

Le modèle gelé en 4-bit prend 4x moins de VRAM qu'en fp16.
Les adapteurs LoRA sont entraînés en bf16 (précision complète pour les gradients).

Pour Qwen2.5-7B avec QLoRA rank=16 :
- VRAM nécessaire : ~8-10 Go
- Une RTX 3060 12 Go peut le faire confortablement

La contrepartie : QLoRA est plus lent qu'un LoRA fp16 pur à cause des opérations de déquantification.
Sur des datasets modestes (< 10k exemples), la différence de qualité finale est souvent marginale.
Sur des datasets larges ou des tâches fines, LoRA fp16 donne de meilleurs résultats.

---

## Tableau de comparaison

| Critère | LoRA fp16 | QLoRA 4-bit |
|---|---|---|
| VRAM requise (7B) | ~18-20 Go | ~8-10 Go |
| Vitesse d'entraînement | Plus rapide | Plus lent (déquantification) |
| Qualité finale | Meilleure | Légèrement inférieure |
| Stabilité | Bonne | Bonne avec NF4 |
| Matériel typique | RTX 3090/4090, A100 | RTX 3060/4060/4070 |

---

## Les modules LoRA ciblés pour Qwen

Pas tous les modules du modèle ne méritent d'être entraînés avec LoRA.
En pratique, on cible les projections d'attention et parfois les projections MLP.

Pour Qwen2.5 :

```
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

Cibler seulement `q_proj`, `v_proj` réduit le nombre de paramètres entraînables
mais donne un signal de gradient moins riche. Pour un dataset > 1000 exemples,
cibler tous les modules d'attention + MLP donne de meilleurs résultats.
