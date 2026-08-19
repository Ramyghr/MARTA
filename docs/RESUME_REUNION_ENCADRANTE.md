# MARTA — Résumé pour la réunion (1 page)

*À utiliser comme support pour présenter le projet en quelques minutes.*

---

## 1. L'objectif du projet (30 secondes)

Construire un système de trading automatique en 4 couches qui travaillent ensemble :

1. **Un module qui apprend une représentation du marché** (réseau de neurones + apprentissage
   contrastif) et détecte des **régimes de marché** (calme, tendance, crise…) via un modèle
   statistique (HMM).
2. **Une équipe de 5 agents LLM qui débattent** (analyse macro, technique, sentiment, risque,
   position contrarian) pour proposer une stratégie, conditionnée par le régime détecté.
3. **Un agent d'apprentissage par renforcement (RL)** qui apprend, par essais successifs, à
   exécuter les meilleures décisions de trading (direction + taille de position).

**Hypothèse de recherche testée :** conditionner le système sur une représentation latente
apprise améliore significativement la performance ajustée au risque, par rapport à des
approches sans cette information de régime.

## 2. Ce qui est fait et fonctionne (fiable, testé)

Les 4 couches sont **construites, intégrées, et validées individuellement** : pipeline de
données (2018–2025, 4 actifs), détection de régime, débat multi-agents, agent RL. Un audit de
fiabilité a permis de trouver et corriger plusieurs bugs subtils qui faussaient les résultats
(non-déterminisme d'évaluation, fuite d'information entre entraînement et validation, données
obsolètes silencieusement réutilisées) — ce travail de rigueur méthodologique est en soi une
contribution du projet.

## 3. Ce qui ne marche pas, et pourquoi (le vrai résultat)

**Le module d'apprentissage de représentation (couche 1) ne produit pas une représentation
utile pour prédire la direction du marché.** Testé et confirmé par un diagnostic complet, puis
par **4 tentatives de ré-entraînement**, chacune ciblant une hypothèse différente (stratégie de
pooling, force de l'augmentation de données, construction des paires positives, exclusion des
faux négatifs) — **aucune ne bat les features brutes (non-apprises)** en termes de pouvoir
prédictif. Une des versions a même été testée dans le pipeline complet (HMM + agents) : les
régimes détectés étaient statistiquement incohérents (quasi aléatoires par rapport aux régimes
de référence).

**Pivot vers les features brutes (48 dernières heures) :** un détecteur de régime basé
directement sur les features brutes est validé (régimes équilibrés, dynamiques,
interprétables). Un agent RL entraîné dessus reste **non rentable**, mais l'écart s'est réduit
significativement après un entraînement plus long (Sharpe de -2.75 à -1.21, plus
statistiquement distinguable du buy-and-hold) — l'agent a appris à réduire son exposition, pas
à prédire la direction du marché (taux de réussite ~47%, proche du hasard).

**Conclusion honnête :** le signal prédictif dans les données brutes existe mais est faible ;
aucune configuration testée ne le transforme en stratégie rentable statistiquement validée.

## 4. Ce qui reste défendable malgré des résultats négatifs

- Système complet et fonctionnel de bout en bout, chaque couche validée individuellement.
- Méthodologie rigoureuse : hypothèses testées une par une, comparées à des baselines fixes,
  résultats négatifs documentés (pas cachés, pas re-testés jusqu'à obtenir un chiffre positif).
- Plusieurs bugs d'ingénierie réels trouvés et corrigés (déterminisme, fuite de données,
  fraîcheur des artefacts) — contribution indépendante du résultat de trading final.

## 5. Questions précises pour vous

1. **Un résultat négatif, bien documenté et méthodologiquement rigoureux, peut-il valider le
   PFE ?**
2. Avec ~2 semaines restantes, **faut-il réduire le périmètre** — défendre le pipeline de
   données + détection de régime + débat multi-agents comme la contribution aboutie, et
   présenter le résultat RL/encodeur comme une investigation à résultat négatif documenté ?
3. **Comment structurer le rapport** pour mettre en avant la démarche scientifique (hypothèses,
   contrôles, résultats négatifs) plutôt que de donner l'impression d'un échec ?
4. Avec le temps restant, **vaut-il mieux consolider/documenter l'existant**, ou tenter une
   dernière piste ciblée ?
5. **Le mentor à l'école ne répond pas** malgré plusieurs relances, et je n'ai pas pu passer ma
   première restitution (crédit B2 non validé) — avez-vous un moyen d'escalader ça
   administrativement ?

---

*Document technique complet (en anglais, avec tous les détails/chiffres) :
`docs/RESULTS_AND_METHODOLOGY.md`. Architecture détaillée du système :
`docs/ARCHITECTURE.md`.*
