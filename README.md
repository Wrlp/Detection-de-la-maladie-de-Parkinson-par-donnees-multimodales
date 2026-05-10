# Detection de la maladie de Parkinson par donnees multimodales

## Objectif
Ce projet analyse les donnees vocales du dataset Parkinsons Telemonitoring (UCI) pour:
- recuperer les donnees,
- verifier leur completude,
- identifier les variables les plus importantes pour expliquer les scores UPDRS.

## Structure du projet
- data/data_voice.py: chargement et verification rapide du dataset.
- analysis/eda_voice.py: audit de qualite des donnees, correlations et importance des variables.
- reports/variables_importantes.md: synthese des variables a privilegier.
- reports/figures/: graphiques generes par l'analyse.

## Installation
1. Creer et activer un environnement virtuel.
2. Installer les dependances:

```bash
pip install -r requirements.txt
```

## Execution
Lancer d'abord le chargement des donnees:

```bash
python data/data_voice.py
```

Puis lancer l'analyse complete:

```bash
python analysis/eda_voice.py
```

## Resultats attendus
Le script d'analyse produit:
- un audit du dataset (dimensions, valeurs manquantes, doublons, types),
- une verification de completude (nombre de lignes completes),
- des visualisations dans reports/figures:
	- correlation_voice.png
	- top_correlations_motor_UPDRS.png
	- top_correlations_total_UPDRS.png
	- feature_importance_motor_UPDRS.png
	- missingness_voice.png (seulement si des valeurs manquantes existent)

## Notes
- Le dataset est recupere depuis UCI via le package ucimlrepo (id=189).
- La selection des variables importantes est documentee dans reports/variables_importantes.md.
- A ce stade, le projet couvre surtout l'audit et l'analyse exploratoire des donnees vocales.
