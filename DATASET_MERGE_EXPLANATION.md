# Fusion des Datasets Vocaux

## Résumé

On a combiné 2 datasets pour avoir plus de données :
- Dataset original : 1 040 samples (520 sains, 520 malades)
- Dataset Oxford : 195 samples (48 sains, 147 malades)
- Résultat : 1 235 samples (568 sains, 667 malades)

---

## Problème

Le dataset original était trop petit et présentait des risques d'overfitting.

---

## Solution

Ajouter un dataset public reconnu (Oxford) pour :
- 195 samples supplémentaires
- Meilleure généralisation du modèle
- Distribution plus réaliste (46% sain / 54% malade)

---

## Processus (5 étapes)

| Étape | Action |
|-------|--------|
| 1 | Charger dataset original + Oxford |
| 2 | Supprimer colonnes inutiles |
| 3 | Harmoniser à 23 features communes |
| 4 | Fusionner : 1 235 samples |
| 5 | Split 80/20 train/test |

---

## Résultats Avant/Après

| Métrique | Avant | Après |
|----------|-------|-------|
| Samples total | 1 040 | 1 235 (+18.8%) |
| Train set | 1 039 | 988 |
| Test set | 167 | 247 |
| Test labels | Absents | Présents |
| Distribution | 50/50 | 46/54 |

---

## Fichiers créés

```
data/voice/
├── train_data_merged.txt   (988 samples)
└── test_data_merged.txt    (247 samples)
```

---

## Utilisation

```python
from voice_loader import load_voice_data

X_train, X_test, y_train, y_test = load_voice_data(use_merged=True)
```

---

## Références

- Dataset Oxford : https://archive.ics.uci.edu/dataset/174/parkinsons
