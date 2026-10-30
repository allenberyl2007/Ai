# Rapport d'audit automatique

Généré par `audit/audit.py`. Bilan : **174 PASS · 17 WARN · 0 FAIL · 12 INFO**.

WARN = point d'attention documenté (pas une erreur bloquante) ; FAIL = à corriger. Les contrôles portent sur la cohérence interne et le comportement des modèles face à la vérité du jumeau numérique : **ce n'est pas une validation sur données réelles** (voir validation_report.md).

## 1 Architecture  (15/16 PASS)

| contrôle | mesuré | attendu | statut | note |
|---|---|---|---|---|
| fichiers requis (main.py + 5 x train.py/predict.py/model.onnx) | True | True | PASS | [] |
| chaque dossier modèle contient UNIQUEMENT train.py, predict.py, model.onnx | True | True | PASS | {'climate': ['__pycache__', 'model.onnx', 'predict.py', 'train.py'], 'soil': ['__pycache__', 'model.onnx', 'predict.py', 'train.py'], 'crop': ['__pycache__', 'model.onnx', 'predict.py', 'train.py'], 'yield': ['__pycache__', 'model.onnx', 'predict.py', 'train.py'], 'recommendation': ['__pycache__', 'model.onnx', 'predict.py', 'train.py']} |
| aucun Docker / Kubernetes / compose | True | True | PASS |  |
| aucun serveur / microservice / API interne (imports) | True | True | PASS | set() |
| une seule base SQLite | True | True | PASS | ['data/simulation.db'] |
| 7 tables requises + datasets | True | True | PASS |  |
| chaque table de données garde sa source | True | True | PASS |  |
| complexité cyclomatique approx. : 3 plus grosses fonctions | core/twin.py:run=75; main.py:recommend=28; data/make_datasets.py:build=17 |  | INFO | twin.run() est un bloc monolithique (boucle journalière) : lisible mais long ; à découper si maintenance difficile |
| complexité max d'une fonction hors twin.run | 28 | [, 60] | PASS |  |
| longueur du plus gros fichier (lignes) | 424 | [, 500] | PASS |  |
| chaque predict.py <= 25 lignes | True | True | PASS |  |
| modèles indépendants (aucun predict.py n'importe un autre modèle) | True | True | PASS |  |
| chaque train.py lit uniquement ses colonnes (SELECT explicite, jamais SELECT *) | True | True | PASS |  |
| aucun train.py ne lit un autre modèle (entraînement indépendant) | True | True | PASS |  |
| docstring de module présente (lisibilité) | True | True | PASS |  |
| fonctionnement HORS LIGNE (réseau bloqué, main.py --demo) | True | True | PASS | culture=maize, confiance=0.68 |

## 2 Données  (35/42 PASS)

| contrôle | mesuré | attendu | statut | note |
|---|---|---|---|---|
| aucun site_id partagé entre train/val/test (ds_main) | True | True | PASS | intersections=0 |
| aucun wseed partagé entre train/val/test (ds_main) | True | True | PASS | intersections=0 |
| aucun pair_id partagé entre train/val/test (ds_main) | True | True | PASS | intersections=0 |
| graines météo disjointes entre splits ET entre ds_climate/ds_main | True | True | PASS |  |
| climate: aucune cible d'un autre modèle parmi ses variables d'entrée | True | True | PASS |  |
| climate: aucune variable « future » (préfixe clim_, cibles) | True | True | PASS |  |
| climate: variables constantes | 0 | [, 0] | PASS |  |
| climate: paires de variables quasi identiques (|corr|>0.98) | 0 | 0 | PASS | [] |
| soil: aucune cible d'un autre modèle parmi ses variables d'entrée | True | True | PASS |  |
| soil: aucune variable « future » (préfixe clim_, cibles) | True | True | PASS |  |
| soil: variables constantes | 0 | [, 0] | PASS |  |
| soil: paires de variables quasi identiques (|corr|>0.98) | 0 | 0 | PASS | [] |
| crop: aucune cible d'un autre modèle parmi ses variables d'entrée | True | True | PASS |  |
| crop: aucune variable « future » (préfixe clim_, cibles) | True | True | PASS |  |
| crop: variables constantes | 0 | [, 0] | PASS |  |
| crop: paires de variables quasi identiques (|corr|>0.98) | 0 | 0 | PASS | [] |
| yield: aucune cible d'un autre modèle parmi ses variables d'entrée | True | True | PASS |  |
| yield: aucune variable « future » (préfixe clim_, cibles) | True | True | PASS |  |
| yield: variables constantes | 0 | [, 0] | PASS |  |
| yield: paires de variables quasi identiques (|corr|>0.98) | 0 | 0 | PASS | [] |
| recommendation: entrées = sorties des 4 autres modèles uniquement (+ culture) | True | True | PASS |  |
| cohérence temporelle : variables pré-saison insensibles à la météo future (test de remplacement) | True | True | PASS |  |
| lignes dupliquées (ds_main) | 0 | [, 0] | PASS |  |
| vecteurs d'entrée de test déjà vus en train (contamination) | 0 | [, 0] | PASS |  |
| unités/plages : valeurs hors plages physiques | 0 | [, 0] | PASS | {} |
| valeurs manquantes (ds_main, ds_climate) | 0, 0 | 0, 0 | PASS |  |
| saisons avec érosion extrême (> 300 t/ha) | 12 | [, 0] | WARN | tempête exceptionnelle sur forte pente : physiquement possible mais rare ; à surveiller |
| valeurs aberrantes yield_t_ha (> 8 MAD, %) | 7.3 | [, 5] | WARN | les queues lourdes sont physiques (érosion, pluies) : conservées, interprétées par les intervalles |
| valeurs aberrantes erosion_t_ha (> 8 MAD, %) | 8.8 | [, 5] | WARN | les queues lourdes sont physiques (érosion, pluies) : conservées, interprétées par les intervalles |
| valeurs aberrantes water_use_mm (> 8 MAD, %) | 2.2 | [, 5] | PASS | les queues lourdes sont physiques (érosion, pluies) : conservées, interprétées par les intervalles |
| valeurs aberrantes season_rain_mm (> 8 MAD, %) | 0.0 | [, 5] | PASS | les queues lourdes sont physiques (érosion, pluies) : conservées, interprétées par les intervalles |
| équilibre des lignes par culture (min/max) | 0.893 | [0.8, ] | PASS |  |
| équilibre des classes de décision (meilleure culture de référence) | maize 59%, bean 18%, cassava 12%, groundnut 10%, rice_upland 1% | chaque classe >= 3 %, aucune > 60 % | WARN | déséquilibre issu de la règle de valeur (policy.py) : le riz pluvial est presque jamais optimal ; ne pas évaluer l'IA par l'exactitude brute mais par le regret |
| régions présentes dans train | 6 | 6 | PASS |  |
| régions présentes dans val | 6 | 6 | PASS |  |
| régions présentes dans test | 6 | 6 | PASS |  |
| équilibre des distributions train vs test : KS max | 0.189 | [, 0.15] | WARN | drain |
| équilibre des distributions train vs test : KS moyen | 0.072 | [, 0.08] | PASS |  |
| part d'événements extrêmes par split | test 29%, train 29%, val 29% | similaire (~30 %) | PASS |  |
| couverture : paires d'entraînement par type d'événement (minimum, hors « none ») | 58 | [50, ] | PASS | {'none': 994, 'dry_spell': 79, 'pest': 71, 'disease': 67, 'flood': 66, 'heatwave': 65, 'drought': 58} |
| couverture : saisons très sèches (pluie < 60 % de la normale, %) | 3.5 | [5, ] | WARN | extrême rare : peu d'exemples pour apprendre la sécheresse |
| couverture : saisons avec au moins 3 jours à 35 °C ou plus (%) | 2.2 | [5, ] | WARN | explique l'absence de gain du Climate AI sur la chaleur |

## 3 Machine learning  (112/133 PASS)

| contrôle | mesuré | attendu | statut | note |
|---|---|---|---|---|
| climate: séparation train/val/test appliquée (tailles 4200/900/900) | True | True | PASS |  |
| climate: parité ONNX vs scikit-learn (écart max) | 1.6e-04 | [, 0.001] | PASS |  |
| climate: stabilité validation croisée par groupes (écart-type R2) | 0.025 | [, 0.1] | PASS |  |
| climate: faux positifs hors-domaine sur test | 0.003 | [, 0.05] | PASS |  |
| climate/season_rain_mm: sur-apprentissage (écart val/train relatif) | 0.128 | [, 0.3] | PASS |  |
| climate/season_rain_mm: R2 train/val/test | 0.27 / 0.20 / 0.15 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| climate/season_rain_mm: couverture de l'intervalle à 80 % sur test | 0.86 | [0.72, 0.88] | PASS |  |
| climate/season_rain_mm: gain vs prédiction constante | 0.10 | [0.05, ] | PASS |  |
| climate/season_rain_mm: courbe d'apprentissage (gain val 50%->100 % des données) | -0.2% | < 5 % = données suffisantes | PASS |  |
| climate/dry_spell_days: sur-apprentissage (écart val/train relatif) | 0.088 | [, 0.3] | PASS |  |
| climate/dry_spell_days: R2 train/val/test | 0.37 / 0.26 / 0.31 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| climate/dry_spell_days: couverture de l'intervalle à 80 % sur test | 0.83 | [0.72, 0.88] | PASS |  |
| climate/dry_spell_days: gain vs prédiction constante | 0.25 | [0.05, ] | PASS |  |
| climate/dry_spell_days: courbe d'apprentissage (gain val 50%->100 % des données) | -0.2% | < 5 % = données suffisantes | PASS |  |
| climate/max3_rain_mm: sur-apprentissage (écart val/train relatif) | 0.111 | [, 0.3] | PASS |  |
| climate/max3_rain_mm: R2 train/val/test | -0.02 / -0.08 / -0.07 | R2 val >= 0.15 (sinon sous-apprentissage) | WARN | peu prévisible avant la saison : l'intervalle porte l'information |
| climate/max3_rain_mm: couverture de l'intervalle à 80 % sur test | 0.82 | [0.72, 0.88] | PASS |  |
| climate/max3_rain_mm: gain vs prédiction constante | 0.01 | [0.05, ] | WARN |  |
| climate/max3_rain_mm: courbe d'apprentissage (gain val 50%->100 % des données) | 1.0% | < 5 % = données suffisantes | PASS |  |
| climate/heat_days: sur-apprentissage (écart val/train relatif) | -2.551 | [, 0.3] | PASS |  |
| climate/heat_days: R2 train/val/test | -0.02 / -0.01 / -0.01 | R2 val >= 0.15 (sinon sous-apprentissage) | WARN | peu prévisible avant la saison : l'intervalle porte l'information |
| climate/heat_days: couverture de l'intervalle à 80 % sur test | 0.98 | [0.72, 0.88] | WARN |  |
| climate/heat_days: gain vs prédiction constante | 0.00 | [0.05, ] | WARN |  |
| climate/heat_days: courbe d'apprentissage (gain val 50%->100 % des données) | 0.0% | < 5 % = données suffisantes | PASS |  |
| climate/et0_season_mm: sur-apprentissage (écart val/train relatif) | 0.099 | [, 0.3] | PASS |  |
| climate/et0_season_mm: R2 train/val/test | 0.47 / 0.40 / 0.40 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| climate/et0_season_mm: couverture de l'intervalle à 80 % sur test | 0.83 | [0.72, 0.88] | PASS |  |
| climate/et0_season_mm: gain vs prédiction constante | 0.23 | [0.05, ] | PASS |  |
| climate/et0_season_mm: courbe d'apprentissage (gain val 50%->100 % des données) | -0.5% | < 5 % = données suffisantes | PASS |  |
| climate: corrections automatiques proposées | max3_rain_mm: biais élevé (R2 train -0.02, val -0.08) -> ajouter des variables informatives ou de la capacité | max3_rain_mm: aucun gain sur la médiane constante (1%) -> variable difficilement prévisible ; la confiance est limitée à l'intervalle | heat_days: biais élevé (R2 train -0.02, val -0.01) -> ajouter des variables informatives ou de la capacité | heat_days: aucun gain sur la médiane consta |  | INFO |  |
| soil: séparation train/val/test appliquée (tailles 6854/1464/1468) | True | True | PASS |  |
| soil: parité ONNX vs scikit-learn (écart max) | 1.7e-04 | [, 0.001] | PASS |  |
| soil: stabilité validation croisée par groupes (écart-type R2) | 0.080 | [, 0.1] | PASS |  |
| soil: faux positifs hors-domaine sur test | 0.029 | [, 0.05] | PASS |  |
| soil/erosion_t_ha: sur-apprentissage (écart val/train relatif) | -0.166 | [, 0.3] | PASS |  |
| soil/erosion_t_ha: R2 train/val/test | 0.36 / 0.28 / 0.42 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| soil/erosion_t_ha: couverture de l'intervalle à 80 % sur test | 0.81 | [0.72, 0.88] | PASS |  |
| soil/erosion_t_ha: gain vs prédiction constante | 0.32 | [0.05, ] | PASS |  |
| soil/erosion_t_ha: courbe d'apprentissage (gain val 50%->100 % des données) | 1.5% | < 5 % = données suffisantes | PASS |  |
| soil/d_soc_pct: sur-apprentissage (écart val/train relatif) | 0.310 | [, 0.3] | WARN |  |
| soil/d_soc_pct: R2 train/val/test | 0.82 / 0.75 / 0.83 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| soil/d_soc_pct: couverture de l'intervalle à 80 % sur test | 0.79 | [0.72, 0.88] | PASS |  |
| soil/d_soc_pct: gain vs prédiction constante | 0.64 | [0.05, ] | PASS |  |
| soil/d_soc_pct: courbe d'apprentissage (gain val 50%->100 % des données) | 2.6% | < 5 % = données suffisantes | PASS |  |
| soil/d_ph: sur-apprentissage (écart val/train relatif) | 0.283 | [, 0.3] | PASS |  |
| soil/d_ph: R2 train/val/test | 0.98 / 0.99 / 0.98 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| soil/d_ph: couverture de l'intervalle à 80 % sur test | 0.81 | [0.72, 0.88] | PASS |  |
| soil/d_ph: gain vs prédiction constante | 0.87 | [0.05, ] | PASS |  |
| soil/d_ph: courbe d'apprentissage (gain val 50%->100 % des données) | 3.0% | < 5 % = données suffisantes | PASS |  |
| soil/n_leach_kg: sur-apprentissage (écart val/train relatif) | 0.265 | [, 0.3] | PASS |  |
| soil/n_leach_kg: R2 train/val/test | 0.91 / 0.89 / 0.88 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| soil/n_leach_kg: couverture de l'intervalle à 80 % sur test | 0.77 | [0.72, 0.88] | PASS |  |
| soil/n_leach_kg: gain vs prédiction constante | 0.68 | [0.05, ] | PASS |  |
| soil/n_leach_kg: courbe d'apprentissage (gain val 50%->100 % des données) | 5.1% | < 5 % = données suffisantes | INFO | plus de données amélioreraient encore le modèle |
| soil: corrections automatiques proposées | d_soc_pct: variance élevée (écart train/val 31%) -> augmenter min_samples_leaf, réduire la profondeur, ajouter des données |  | INFO |  |
| crop: séparation train/val/test appliquée (tailles 6854/1464/1468) | True | True | PASS |  |
| crop: parité ONNX vs scikit-learn (écart max) | 6.1e-04 | [, 0.001] | PASS |  |
| crop: stabilité validation croisée par groupes (écart-type R2) | 0.000 | [, 0.1] | PASS |  |
| crop: faux positifs hors-domaine sur test | 0.022 | [, 0.05] | PASS |  |
| crop/days_to_harvest: sur-apprentissage (écart val/train relatif) | 0.148 | [, 0.3] | PASS |  |
| crop/days_to_harvest: R2 train/val/test | 1.00 / 1.00 / 1.00 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| crop/days_to_harvest: couverture de l'intervalle à 80 % sur test | 0.79 | [0.72, 0.88] | PASS |  |
| crop/days_to_harvest: gain vs prédiction constante | 0.96 | [0.05, ] | PASS |  |
| crop/days_to_harvest: courbe d'apprentissage (gain val 50%->100 % des données) | 0.2% | < 5 % = données suffisantes | PASS |  |
| crop/water_use_mm: sur-apprentissage (écart val/train relatif) | 0.184 | [, 0.3] | PASS |  |
| crop/water_use_mm: R2 train/val/test | 0.98 / 0.97 / 0.97 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| crop/water_use_mm: couverture de l'intervalle à 80 % sur test | 0.80 | [0.72, 0.88] | PASS |  |
| crop/water_use_mm: gain vs prédiction constante | 0.84 | [0.05, ] | PASS |  |
| crop/water_use_mm: courbe d'apprentissage (gain val 50%->100 % des données) | 1.3% | < 5 % = données suffisantes | PASS |  |
| crop/water_stress: sur-apprentissage (écart val/train relatif) | 0.204 | [, 0.3] | PASS |  |
| crop/water_stress: R2 train/val/test | 0.77 / 0.73 / 0.75 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| crop/water_stress: couverture de l'intervalle à 80 % sur test | 0.87 | [0.72, 0.88] | PASS |  |
| crop/water_stress: gain vs prédiction constante | 0.52 | [0.05, ] | PASS |  |
| crop/water_stress: courbe d'apprentissage (gain val 50%->100 % des données) | 4.7% | < 5 % = données suffisantes | PASS |  |
| crop/n_uptake_kg: sur-apprentissage (écart val/train relatif) | 0.268 | [, 0.3] | PASS |  |
| crop/n_uptake_kg: R2 train/val/test | 0.92 / 0.88 / 0.89 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| crop/n_uptake_kg: couverture de l'intervalle à 80 % sur test | 0.86 | [0.72, 0.88] | PASS |  |
| crop/n_uptake_kg: gain vs prédiction constante | 0.70 | [0.05, ] | PASS |  |
| crop/n_uptake_kg: courbe d'apprentissage (gain val 50%->100 % des données) | 5.8% | < 5 % = données suffisantes | INFO | plus de données amélioreraient encore le modèle |
| crop: corrections automatiques proposées | aucune correction nécessaire selon les seuils automatiques |  | INFO |  |
| yield: séparation train/val/test appliquée (tailles 6854/1464/1468) | True | True | PASS |  |
| yield: parité ONNX vs scikit-learn (écart max) | 9.2e-06 | [, 0.001] | PASS |  |
| yield: stabilité validation croisée par groupes (écart-type R2) | 0.010 | [, 0.1] | PASS |  |
| yield: faux positifs hors-domaine sur test | 0.025 | [, 0.05] | PASS |  |
| yield/yield_t_ha: sur-apprentissage (écart val/train relatif) | 0.161 | [, 0.3] | PASS |  |
| yield/yield_t_ha: R2 train/val/test | 0.90 / 0.88 / 0.86 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| yield/yield_t_ha: couverture de l'intervalle à 80 % sur test | 0.82 | [0.72, 0.88] | PASS |  |
| yield/yield_t_ha: gain vs prédiction constante | 0.68 | [0.05, ] | PASS |  |
| yield/yield_t_ha: courbe d'apprentissage (gain val 50%->100 % des données) | 5.9% | < 5 % = données suffisantes | INFO | plus de données amélioreraient encore le modèle |
| yield: corrections automatiques proposées | aucune correction nécessaire selon les seuils automatiques |  | INFO |  |
| recommendation: séparation train/val/test appliquée (tailles 6854/1464/1468) | True | True | PASS |  |
| recommendation: parité ONNX vs scikit-learn (écart max) | 5.7e-06 | [, 0.001] | PASS |  |
| recommendation: stabilité validation croisée par groupes (écart-type R2) | 0.001 | [, 0.1] | PASS |  |
| recommendation: faux positifs hors-domaine sur test | 0.005 | [, 0.05] | PASS |  |
| recommendation/score_ref: sur-apprentissage (écart val/train relatif) | 0.255 | [, 0.3] | PASS |  |
| recommendation/score_ref: R2 train/val/test | 0.99 / 0.98 / 0.99 | R2 val >= 0.15 (sinon sous-apprentissage) | PASS |  |
| recommendation/score_ref: couverture de l'intervalle à 80 % sur test | 0.80 | [0.72, 0.88] | PASS |  |
| recommendation/score_ref: gain vs prédiction constante | 0.92 | [0.05, ] | PASS |  |
| recommendation/score_ref: courbe d'apprentissage (gain val 50%->100 % des données) | 8.3% | < 5 % = données suffisantes | INFO | plus de données amélioreraient encore le modèle |
| recommendation: corrections automatiques proposées | aucune correction nécessaire selon les seuils automatiques |  | INFO |  |
| soil: robustesse au bruit +-5 % sur les entrées (variation relative moyenne des sorties) | 0.050 | [, 0.1] | PASS |  |
| soil: détection hors-domaine (décalage de 8 écarts-types sur 'soc_pct') | True | True | PASS | détecté 100% |
| soil: baisse de confiance sur données hors domaine | 0.21 | [, 0.6] | PASS |  |
| crop: robustesse au bruit +-5 % sur les entrées (variation relative moyenne des sorties) | 0.075 | [, 0.1] | PASS |  |
| crop: détection hors-domaine (décalage de 8 écarts-types sur 'lat') | True | True | PASS | détecté 100% |
| crop: baisse de confiance sur données hors domaine | 0.22 | [, 0.6] | PASS |  |
| yield: robustesse au bruit +-5 % sur les entrées (variation relative moyenne des sorties) | 0.051 | [, 0.1] | PASS |  |
| yield: détection hors-domaine (décalage de 8 écarts-types sur 'lat') | True | True | PASS | détecté 100% |
| yield: baisse de confiance sur données hors domaine | 0.28 | [, 0.6] | PASS |  |
| Yield AI : région 'bukavu' jamais vue (R2) vs région vue | 0.54 vs 0.76 | chute attendue | INFO | mesure de la généralisation spatiale ; si la chute est forte, le système doit signaler « données jamais vues » |
| Yield AI : région 'lubumbashi' jamais vue (R2) vs région vue | 0.45 vs 0.72 | chute attendue | INFO | mesure de la généralisation spatiale ; si la chute est forte, le système doit signaler « données jamais vues » |
| Yield AI par event = disease (n=44) : couverture 80 % / R2 / MAE t/ha | 86% / 0.81 / 0.76 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par event = drought (n=60) : couverture 80 % / R2 / MAE t/ha | 68% / 0.72 / 1.23 | couverture 72-88 %, R2 >= 0.6 | WARN | trop sûr de lui : intervalle trop étroit |
| Yield AI par event = dry_spell (n=61) : couverture 80 % / R2 / MAE t/ha | 82% / 0.89 / 0.54 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par event = flood (n=69) : couverture 80 % / R2 / MAE t/ha | 83% / 0.86 / 0.49 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par event = heatwave (n=104) : couverture 80 % / R2 / MAE t/ha | 88% / 0.87 / 0.79 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par event = none (n=1036) : couverture 80 % / R2 / MAE t/ha | 83% / 0.88 / 0.63 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par event = pest (n=94) : couverture 80 % / R2 / MAE t/ha | 81% / 0.75 / 0.80 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par region = bukavu (n=235) : couverture 80 % / R2 / MAE t/ha | 74% / 0.76 / 1.40 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par region = kananga (n=220) : couverture 80 % / R2 / MAE t/ha | 75% / 0.70 / 0.62 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par region = kinshasa (n=255) : couverture 80 % / R2 / MAE t/ha | 85% / 0.83 / 0.37 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par region = kisangani (n=265) : couverture 80 % / R2 / MAE t/ha | 86% / 0.87 / 0.74 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par region = lubumbashi (n=218) : couverture 80 % / R2 / MAE t/ha | 84% / 0.72 / 0.28 | couverture 72-88 %, R2 >= 0.6 | PASS |  |
| Yield AI par region = mbandaka (n=275) : couverture 80 % / R2 / MAE t/ha | 89% / 0.94 / 0.61 | couverture 72-88 %, R2 >= 0.6 | WARN | intervalle trop large (prudent, sans danger) |
| calibration P(saison sèche >14 j) : ECE | 0.067 | [, 0.1] | PASS | Brier 0.142 vs climatologie 0.199 |
| calibration P(pluie 3 j >100 mm) : ECE | 0.038 | [, 0.1] | PASS | Brier 0.226 vs climatologie 0.225 |
| BOUT EN BOUT (300 paires de test) : regret moyen relatif IA / aléatoire / toujours maïs / meilleure culture par région | 6.1% / 33.4% / 14.6% / 13.0% | IA < toutes les références | PASS | regret = score de la meilleure culture (vérité du jumeau) - score de la culture recommandée |
| bout en bout : la culture recommandée est la meilleure (exactitude top-1) | 0.67 | [0.4, ] | PASS | pas de chance = 20 % ; classes déséquilibrées |
| bout en bout : recommandation à moins de 10 % du meilleur score | 0.76 | [0.6, ] | PASS |  |
| bout en bout : couverture de l'intervalle du score (80 % nominal) | 0.26 | [0.7, 0.9] | WARN | chaîne réelle : les erreurs des modèles amont se cumulent |
| confiance affichée vs réussite réelle (tercile bas/moyen/haut) | basse 71%, moyenne 74%, haute 82% | croissante | PASS | confiance moyenne 47% ; réussite moyenne 76% |
| confiance moyenne - réussite moyenne (sur-confiance si > 0) | -0.29 | [-0.2, 0.1] | WARN | la confiance ne doit pas dépasser la réussite observée |
| confiance maximale affichée (jamais de certitude) | 0.69 | [, 0.95] | PASS |  |

## 4 Jumeau numérique  (9/9 PASS)

| contrôle | mesuré | attendu | statut | note |
|---|---|---|---|---|
| audit du jumeau — 1 Climat | 54/54 PASS (INFO inclus) | aucun FAIL | PASS |  |
| audit du jumeau — 2 Bilans (conservation) | 5/5 PASS (INFO inclus) | aucun FAIL | PASS |  |
| audit du jumeau — 3 Rendements (plausibilité) | 11/11 PASS (INFO inclus) | aucun FAIL | PASS |  |
| audit du jumeau — 4 Sensibilités (réponse causale) | 11/11 PASS (INFO inclus) | aucun FAIL | PASS |  |
| audit du jumeau — 5 Dynamique du sol (10 ans) | 4/4 PASS (INFO inclus) | aucun FAIL | PASS |  |
| audit du jumeau — 6 Robustesse des entrées | 5/5 PASS (INFO inclus) | aucun FAIL | PASS |  |
| audit du jumeau — 8 Performance | 4/4 PASS (INFO inclus) | aucun FAIL | PASS |  |
| audit du jumeau — 9 Stabilité numérique | 3/3 PASS (INFO inclus) | aucun FAIL | PASS |  |
| adaptateurs (POWER, GPM, SMAP, MODIS, SoilGrids, FAOSTAT) sur charges fictives | 18/18 tests adaptateurs OK | tous OK | PASS | format documenté, JAMAIS testé contre les services réels |

## 5 Validation scientifique  (3/3 PASS)

| contrôle | mesuré | attendu | statut | note |
|---|---|---|---|---|
| aucune validation externe déclarée sans données réelles (GPM/SMAP/MODIS : « NON RÉALISÉE ») | True | True | PASS | lignes réelles en base : 0 |
| code de validation exécutable sur données factices (test du code uniquement) | True | True | PASS | ce test ne valide PAS le jumeau |
| rapport de validation (reports/validation_report.md) | généré |  | PASS | contenu : validation interne / calibration / validation externe (externe = NON RÉALISÉE sans données réelles) |
