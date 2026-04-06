# 02 — Préparation du dataset

## Pourquoi la qualité du dataset prime sur tout le reste

La règle empirique en fine-tuning supervisé : garbage in, garbage out.
Un modèle entraîné sur 200 exemples bien construits surpasse presque toujours
un modèle entraîné sur 5000 exemples bâclés.

Ce qui définit un bon exemple :
- L'instruction est claire et non ambiguë
- La réponse est exactement ce qu'on veut que le modèle produise
- Il n'y a pas de contenu parasite (disclaimers automatiques, répétitions de l'instruction, remplissage)
- La réponse est cohérente avec les autres exemples du dataset

---

## Formats de dataset

### Format Alpaca

Le format le plus simple. Trois champs : instruction, input, output.

```json
{
  "instruction": "Résume ce texte en une phrase.",
  "input": "Le fine-tuning consiste à...",
  "output": "Le fine-tuning adapte un modèle pré-entraîné à une tâche spécifique."
}
```

`input` peut être vide si l'instruction se suffit à elle-même.
C'est le format le plus facile à préparer manuellement.

### Format ShareGPT

Adapté aux conversations multi-tours. Chaque exemple est une conversation complète.

```json
{
  "conversations": [
    {"from": "human", "value": "Explique-moi LoRA en deux lignes."},
    {"from": "gpt", "value": "LoRA entraîne de petites matrices additionnelles..."},
    {"from": "human", "value": "Et QLoRA ?"},
    {"from": "gpt", "value": "QLoRA fait la même chose mais sur un modèle quantifié en 4-bit."}
  ]
}
```

Utile pour fine-tuner un modèle de chat qui doit maintenir le contexte sur plusieurs tours.
Plus complexe à préparer mais plus réaliste si l'usage cible est conversationnel.

### Format ChatML

Le format natif de Qwen. C'est ce que le tokenizer applique en interne
quand on utilise `apply_chat_template`.

```
<|im_start|>system
Tu es un assistant technique spécialisé en fine-tuning de LLMs.<|im_end|>
<|im_start|>user
Explique LoRA.<|im_end|>
<|im_start|>assistant
LoRA (Low-Rank Adaptation) est une technique...<|im_end|>
```

En pratique : préparer les données en Alpaca ou ShareGPT, et laisser `apply_chat_template`
faire la conversion automatiquement au moment de la tokenisation. Ne pas construire
les templates ChatML à la main — le tokenizer s'en charge mieux que toi.

---

## Taille minimale recommandée

Il n'y a pas de seuil universel. Quelques repères :

- **50-100 exemples** : suffisant pour adapter le *style* ou le *format* si les exemples sont très homogènes
- **200-500 exemples** : bon équilibre pour une tâche précise et bien définie
- **1000-5000 exemples** : pour une tâche avec de la variance (plusieurs styles de réponse, plusieurs types d'entrée)
- **> 10 000 exemples** : territoire du fine-tuning de capacité (enseigner quelque chose de nouveau)

Pour Qwen2.5-7B-Instruct sur une tâche d'adaptation de style ou de format :
300 exemples bien construits sont suffisants pour obtenir un comportement stable.

---

## Nettoyage des données

Les problèmes les plus courants dans les datasets bruts :

**Exemples trop courts** : une réponse de 3 mots n'apprend rien au modèle.
Filtrer les réponses en dessous d'un seuil (ex : 20 tokens minimum).

**Exemples trop longs** : les exemples qui dépassent la longueur de contexte max
sont tronqués par le tokenizer. Vérifier la distribution des longueurs
et filtrer ou résumer les exemples hors limites.

**Doublons** : des exemples identiques ou quasi-identiques biaisent l'entraînement
vers ces cas. Dédupliquer sur le champ `instruction` au minimum.

**Incohérences de style** : si certains exemples finissent par un point
et d'autres non, si certains tutoient et d'autres vouvoient, le modèle apprend
un comportement instable. Uniformiser avant d'entraîner.

**Hallucinations dans les réponses** : si le dataset a été généré par un autre LLM,
relire un échantillon manuellement. Les LLMs génèrent des datasets plausibles
mais parfois factuellement faux. Si la tâche exige de la précision factuelle,
chaque exemple doit être vérifié.

---

## Ce qui fait un bon exemple d'entraînement

Un bon exemple montre exactement le comportement cible, sans ambiguité.

Mauvais exemple :
```json
{
  "instruction": "Aide-moi avec du code Python.",
  "input": "def foo(x): return x * 2",
  "output": "Votre code est correct. Vous pouvez aussi écrire foo = lambda x: x * 2."
}
```

Pourquoi c'est mauvais : "Aide-moi" est vague, la réponse commence par
une validation inutile, la suggestion lambda n'était pas demandée.

Bon exemple :
```json
{
  "instruction": "Explique ce que fait cette fonction Python.",
  "input": "def foo(x): return x * 2",
  "output": "Cette fonction prend un nombre x en entrée et retourne sa valeur multipliée par 2."
}
```

Pourquoi c'est bon : instruction précise, réponse directe, pas de remplissage.

---

## Script de préparation

Le script `scripts/prepare_dataset.py` prend un fichier JSONL brut
(un objet JSON par ligne) et produit un fichier JSON au format Alpaca
avec les filtrages basiques appliqués.

Format attendu en entrée (flexible — le script mappe les champs) :

```jsonl
{"question": "...", "answer": "..."}
{"prompt": "...", "response": "..."}
{"instruction": "...", "output": "..."}
```

Voir les options de mapping dans le script.
