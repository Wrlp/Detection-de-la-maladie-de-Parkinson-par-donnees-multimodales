# Résumé des variables importantes

Ce résumé se base sur l’analyse du dataset **Parkinsons Telemonitoring** utilisé dans la partie audio.

## Variables à garder en priorité

### Variables vocales les plus informatives
- `HNR` : ressort souvent parmi les variables les plus liées aux scores UPDRS.
- `RPDE` : mesure la complexité du signal vocal, utile pour capter des irrégularités liées à la maladie.
- `DFA` : décrit l’aspect fractal du signal, donc une information complémentaire sur la structure de la voix.
- `PPE` : indicateur non linéaire pertinent pour la variation de la fréquence fondamentale.
- `Shimmer(dB)` : mesure les variations d’amplitude, souvent utiles pour détecter une altération vocale.
- `Shimmer:APQ11` et `Shimmer:APQ5` : versions plus robustes des mesures de shimmer.
- `Jitter(%)` et `Jitter(Abs)` : utiles pour quantifier l’instabilité de la fréquence fondamentale.

### Variables de contexte à conserver
- `age` : influence visible dans l’analyse et utile comme variable explicative.
- `sex` : peut apporter un signal complémentaire.
- `test_time` : utile car l’évolution dans le temps est importante dans le suivi médical.

## Variables à éviter comme entrée du modèle
- `motor_UPDRS` et `total_UPDRS` ne doivent pas être utilisées comme variables d’entrée si l’une d’elles est la cible.
- Elles sont très fortement corrélées entre elles, donc il faut éviter la fuite de données.

## Variables moins prioritaires
- Les variables de type `Shimmer` et `Jitter` très proches entre elles sont souvent redondantes.
- Il vaut mieux garder quelques indicateurs représentatifs plutôt que toutes les variantes si le modèle devient trop complexe.

## Conclusion
Pour la partie audio, il est raisonnable de conserver en priorité les mesures vocales suivantes : `HNR`, `RPDE`, `DFA`, `PPE`, `Shimmer(dB)`, `Shimmer:APQ11`, `Shimmer:APQ5`, `Jitter(%)`, `Jitter(Abs)`, ainsi que `age`, `sex` et `test_time`.

La logique pour la partie accéléromètre sera la même : garder les variables les plus corrélées à la cible, supprimer les redondances et justifier le choix avec des visualisations.
