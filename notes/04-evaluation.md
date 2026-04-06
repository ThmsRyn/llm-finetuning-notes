# 04 — Evaluation d'un modèle fine-tuné

## Pourquoi l'évaluation est souvent bâclée

L'entraînement est excitant. L'évaluation l'est moins.
Résultat : la plupart des expériences de fine-tuning se terminent par un "ça a l'air mieux"
après deux ou trois prompts manuels, sans méthodologie structurée.

Le problème : sans évaluation rigoureuse, on ne sait pas si le modèle est réellement meilleur,
si les améliorations tiennent sur des cas réels, ou si on a juste réduit la variance sur les cas de test.

---

## Perplexité — ce que ca mesure réellement

La perplexité mesure à quel point le modèle est "surpris" par le texte.
Une perplexité basse = le modèle prédit bien les tokens suivants.
C'est la métrique automatique la plus facile à calculer.

Formule : perplexité = exp(cross-entropy loss)

Une training loss basse sans descente de la validation loss = overfitting.
Une validation loss qui descend avec la training loss = le modèle généralise.

**Ce que la perplexité ne mesure pas** : la qualité réelle des réponses.
Un modèle peut avoir une perplexité très basse sur le dataset de validation
et quand même produire des réponses inutiles sur des cas non vus.
La perplexité est un signal nécessaire, pas suffisant.

### Comment la calculer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

model = AutoModelForCausalLM.from_pretrained("chemin/vers/modele")
tokenizer = AutoTokenizer.from_pretrained("chemin/vers/modele")

texte = "Un exemple de texte pour tester."
inputs = tokenizer(texte, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

perplexite = math.exp(loss.item())
print(f"Perplexité : {perplexite:.2f}")
```

---

## Evaluation manuelle — la seule qui compte vraiment

Créer un set d'évaluation fixe : 20 à 50 prompts représentatifs de l'usage cible,
avec la réponse attendue ou des critères d'évaluation clairs.

Grille simple pour chaque réponse :

| Critère | Score 1-3 |
|---|---|
| La réponse répond à la question posée | 1 = non / 2 = partiellement / 3 = oui |
| Le format est correct (longueur, structure) | 1 = non / 2 = partiellement / 3 = oui |
| Le style correspond au comportement cible | 1 = non / 2 = partiellement / 3 = oui |
| Absence d'hallucination ou d'information fausse | 1 = présente / 2 = incertaine / 3 = aucune |

Faire passer ce set sur le modèle de base et sur le modèle fine-tuné.
Comparer les scores. Si le score global n'est pas meilleur, l'entraînement
n'a pas atteint son objectif.

---

## Comparaison avant/après

Protocole minimal :

1. Garder 10-15% des exemples du dataset hors de l'entraînement (split eval)
2. Faire tourner le modèle de base sur ces exemples
3. Faire tourner le modèle fine-tuné sur ces mêmes exemples
4. Comparer côte à côte, pas de façon globale

Le split eval doit être représentatif mais ne doit pas contenir les exemples
les plus faciles (sinon les deux modèles s'en sortent bien et la comparaison
ne révèle rien).

---

## Overfitting — comment le détecter

Signaux d'overfitting :
- La training loss descend, la validation loss remonte ou se stabilise puis monte
- Le modèle reproduit mot pour mot des phrases des exemples d'entraînement
- Le modèle répond bien aux prompts proches du dataset mais dégrade sur les variations

Solutions :
- Réduire le nombre d'epochs (souvent 1-2 suffisent sur petits datasets)
- Augmenter le dropout LoRA (`lora_dropout=0.1`)
- Diversifier le dataset (moins de doublons ou quasi-doublons)
- Activer `weight_decay=0.01` dans les TrainingArguments

---

## Catastrophic forgetting

Le catastrophic forgetting est la dégradation des capacités générales du modèle
de base au profit de la tâche fine-tunée. En LoRA, ce problème est fortement atténué
parce qu'on ne touche pas aux poids de base — on ajoute des couches.

Mais il peut quand même apparaître si :
- Le learning rate est trop élevé (> 5e-4)
- Le dataset est trop homogène et très différent de la distribution pré-entraînement
- On entraîne trop longtemps (trop d'epochs)

Comment le détecter : tester le modèle fine-tuné sur des tâches générales
(résumé, traduction, code) non liées à la tâche cible. Si les réponses dégradent
nettement par rapport au modèle de base, il y a du forgetting.

Correction : réduire le learning rate, le nombre d'epochs, ou l'alpha LoRA.

---

## Pièges classiques à éviter

**Evaluer uniquement sur les exemples d'entraînement** : le modèle les a mémorisés,
les bons scores ne signifient rien.

**Comparer sur un seul prompt** : un prompt peut être flatteur pour le modèle fine-tuné
par hasard. Toujours comparer sur un ensemble représentatif.

**Ignorer les cas limites** : tester aussi sur des prompts hors distribution, ambigus,
ou incorrectement formulés. Un bon fine-tuning doit rester robuste, pas juste mémoriser.

**Confondre "plus verbeux" avec "meilleur"** : les modèles fine-tunés ont tendance
à produire des réponses plus longues si les exemples d'entraînement sont longs.
Longueur != qualité.

**Ne pas versionner les checkpoints** : toujours garder le checkpoint intermédiaire
le plus performant, pas seulement le dernier. `load_best_model_at_end=True` fait ca automatiquement.
