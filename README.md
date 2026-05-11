# Detection de la maladie de Parkinson par donnees multimodales

## Membres de l'équipe
> Flavien Baron & Ewan Schwaller & Anna-Eve Mercier & Guillermo Milne & Laure Warlop

## Objectif
Ce projet detecte la maladie de Parkinson a partir de donnees multimodales :
- donnees vocales (enregistrements de la voix)
- donnees de spirales dessinees sur tablette graphique (tremblements)

Un modele de fusion tardive combine les deux sources pour produire une prediction finale.

## Structure du projet
```
├── analysis/
│   ├── eda_spiral_uci.py        # audit qualite + features spirales (UCI 395)
│   └── eda_voice.py             # audit qualite + figures vocales
├── data/
│   ├── raw/spiral_uci/          # donnees brutes spirales (UCI 395)
│   ├── voice/
│   │   ├── train_data.txt
│   │   ├── test_data.txt
│   │   ├── train_data_merged.txt
│   │   └── test_data_merged.txt
│   └── data_voice.py            # verification rapide du dataset vocal
├── model/
│   └── fusion_model.py          # modele de fusion tardive (RandomForest x2)
├── reports/
│   ├── figures/
│   │   ├── spiral_uci/          # features_per_subject.csv + figures EDA spirales
│   │   ├── voice/               # figures EDA vocales + CSV agrege
│   │   ├── confusion_matrix_fusion.png
│   │   ├── feature_importance_spiral_model.png
│   │   ├── feature_importance_voice_model.png
│   │   ├── metrics_baseline.json
│   │   └── roc_curves_comparison.png
│   └── variables_importantes.md
├── .gitignore
├── DATASET_MERGE_EXPLANATION.md
├── pyproject.toml
├── README.md
├── requirements.txt
├── uv.lock
└── voice_loader.py              # chargement des donnees vocales (train/test .txt)
```

## Installation
1. Creer et activer un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Installer les dependances :
```bash
pip install -r requirements.txt
```

## Execution
### 1. Analyse exploratoire des donnees spirales
Produit `reports/figures/spiral_uci/features_per_subject.csv` necessaire pour le modele :
```bash
python analysis/eda_spiral_uci.py
```

### 2. Analyse exploratoire des donnees vocales
Produit les figures d'analyse et un CSV agrege par sujet :
```bash
python analysis/eda_voice.py
```

### 3. Modele de fusion
Charge les deux sources, entraine les modeles et produit les figures de resultats :
```bash
python model/fusion_model.py
```

## Resultats attendus
### Analyse vocale (`analysis/eda_voice.py`)
- audit du dataset (dimensions, valeurs manquantes, doublons, types)
- verification de completude
- visualisations dans `reports/figures/voice/` :
  - `correlation_voice.png`
  - `top_correlations_motor_UPDRS.png`
  - `top_correlations_total_UPDRS.png`
  - `feature_importance_motor_UPDRS.png`
  - `missingness_voice.png` (seulement si des valeurs manquantes existent)

### Analyse spirales (`analysis/eda_spiral_uci.py`)
- features agregees par sujet dans `reports/figures/spiral_uci/features_per_subject.csv`
- visualisations : effectifs, completude des epreuves, boxplots, correlations, importance RF

### Modele de fusion (`model/fusion_model.py`)
- deux RandomForestClassifier independants (voix + spirale)
- fusion tardive par concatenation des probabilites
- sorties dans `reports/figures/` :
  - `confusion_matrix_fusion.png`
  - `roc_curves_comparison.png`
  - `feature_importance_voice_model.png`
  - `feature_importance_spiral_model.png`
  - `metrics_baseline.json` (AUC par modalite + fusion, pour le tuning hyperparametres)

## Performances baseline
| Modalite | AUC (cross-val) | AUC (holdout) |
|----------|----------------|---------------|
| Voix     | 0.999 ± 0.002  | 1.000         |
| Spirale  | 0.956 ± 0.036  | 0.922         |
| Fusion   | /              | 0.999         |

## Notes
- Le dataset vocal utilise `voice_loader.py` avec le dataset fusionne (`use_merged=True`).
- Le dataset spirale est recupere depuis UCI 395 via `eda_spiral_uci.py`.
- La selection des variables importantes est documentee dans `reports/variables_importantes.md`.
- La normalisation est geree par le pipeline scikit-learn (`StandardScaler`) et non par `voice_loader`.
- `metrics_baseline.json` sert de reference pour la partie optimisation des hyperparametres.

