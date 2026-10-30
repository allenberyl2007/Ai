# Aide à la décision agricole — RDC (NASA Space Apps)

Système local et hors ligne : **un jumeau numérique, une seule base SQLite (`data/simulation.db`), cinq modèles IA, un seul point d'entrée (`main.py`)**.
Aucun serveur, aucune API interne, aucun Docker/Kubernetes.

```
python main.py --region kananga --soil ferralsol_scl --slope 8 --prev bean --demo
```
Sortie (exemple de format) : *Culture recommandée / Pourquoi ? / Récolte attendue / Confiance / Alternative*.

## Structure
```
main.py                      point d'entrée : Climate AI -> Soil AI -> Crop AI -> Yield AI -> Recommendation AI (appels successifs)
core/                        jumeau numérique (twin, climate, crops, regions, events, features, schema) + db + mlkit (+ policy, decide, explain)
models/<modèle>/             climate, soil, crop, yield, recommendation : UNIQUEMENT train.py, predict.py, model.onnx
adapters/                    un adaptateur par source : power, smap, gpm, modis, soilgrids, faostat
data/make_datasets.py        le jumeau numérique génère les datasets (tables ds_* de simulation.db)
scripts/                     create_db.py, download_all.py, train_all.sh
audit/                       audit.py (audit automatique), audit_twin.py (jumeau), validate.py (validation scientifique)
tests/test_adapters.py       tests des adaptateurs sur charges fictives
reports/                     audit_report.md, validation_report.md, twin_checks.md
```

## Base SQLite unique
Tables : `weather`, `soil`, `soil_moisture`, `vegetation`, `yield_reference`, `crop`, `recommendation` — chacune a une colonne `source`
(`sim`, `archetype`, `nasa_power`, `gpm`, `smap`, `modis`, `soilgrids`, `faostat`, `literature`).
Les datasets d'entraînement produits par le jumeau (`ds_main`, `ds_climate`) sont rangés dans la **même** base (interprétation de « une seule base »).
Le dossier de chaque modèle ne contient que 3 fichiers : les métadonnées (variables, diagnostics, calibration, référence hors-domaine) sont **embarquées dans `model.onnx`**.

## Reconstruire de zéro
```
python scripts/create_db.py                         # base + archétypes de sols + paramètres des cultures
python data/make_datasets.py                        # le jumeau génère les datasets (~6 min, 1 cœur)
bash scripts/train_all.sh                           # 5 modèles (train.py indépendants)
python audit/audit_twin.py && python audit/audit.py # audits -> reports/
```

## Passer aux données réelles (sans changer l'architecture)
1. `python scripts/download_all.py kananga 2015-01-01 2024-12-31` (Internet ; SMAP/MODIS : `EARTHDATA_USER`, `EARTHDATA_PASS`) remplit la base.
2. `python audit/validate.py` produit calibration et validation externe (« NON RÉALISÉE » tant que les lignes réelles manquent).
3. Ajuster `core/regions.py` (normales) selon le rapport de calibration, relancer `make_datasets.py` puis `train_all.sh`.
4. `python main.py --mode real ...` lit en priorité `nasa_power` / `soilgrids`, puis le simulé.

## Importer des fichiers téléchargés à la main (sans API)
```
python scripts/import_csv.py power pt_0.2987_23.9756 POWER_Point_Monthly_....csv   # POWER mensuel : calibration seulement (pas lu par le jumeau)
python scripts/import_csv.py power kananga POWER_Daily_....csv                     # POWER journalier : 7 paramètres exigés (PRECTOTCORR, T2M, T2M_MAX, T2M_MIN, ALLSKY_SFC_SW_DWN, RH2M, WS2M)
python scripts/import_csv.py faostat FAOSTAT_data_en_....csv                       # export FAOSTAT (kg/ha ou hg/ha), drapeaux de qualité conservés
```
Le jumeau ne lit que la météo **journalière** (sources `nasa_power`, `sim`). Les fichiers mensuels servent à la calibration (rapport de validation).

## Incertitude et confiance
Chaque modèle renvoie, pour chaque cible : prédiction (`_mid`), intervalle à 80 % (`_lo`, `_hi`, calibré par prédiction conforme), incertitude (`_unc`), confiance (`_conf`),
et signale `ood_flag` / `unseen_features` (données jamais vues / hors domaine). Plus l'intervalle est large ou la situation inhabituelle, plus la confiance baisse.
Confiance finale = min des confiances des modèles × probabilité que la 1re culture batte la 2e ; plafonnée à 85 % tant qu'aucune donnée réelle n'a servi (95 % au maximum ensuite).

## Limites à connaître
- Tous les modèles sont entraînés sur des **simulations** : leurs scores mesurent l'accord avec le jumeau, pas avec la réalité.
- La règle de valeur d'une culture (`core/policy.py` : calories/ha/100 jours, pénalités sol et eau) est un choix de conception modifiable ; aucun prix de marché n'est utilisé.
- Les adaptateurs n'ont **jamais été testés contre les services réels** (réseau désactivé à la construction) ; leurs `parse()` sont testés sur des charges fictives.
- Voir `reports/audit_report.md` et `reports/validation_report.md` pour les mesures et les points d'attention.
