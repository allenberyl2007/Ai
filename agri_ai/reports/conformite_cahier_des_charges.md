# Conformité au cahier des charges

| Exigence | Statut | Où / remarque |
|---|---|---|
| Jumeau numérique + une seule base SQLite + 5 modèles + `main.py` | Conforme | `core/` (jumeau), `data/simulation.db`, `models/`, `main.py` |
| Aucun microservice / API interne / Docker / Kubernetes / serveur | Conforme | vérifié automatiquement (imports, fichiers) — audit §1 |
| Fonctionne hors ligne | Conforme | audit §1 : `main.py --demo` exécuté avec le réseau bloqué |
| Sources : POWER, SMAP, GPM, MODIS NDVI, SoilGrids, FAOSTAT, un adaptateur chacun, qui télécharge puis remplit SQLite | Conforme en structure ; **jamais testé sur les vrais services** | `adapters/*.py` ; `parse()` testés sur charges fictives (18/18) ; SMAP/MODIS exigent un compte Earthdata |
| Tables weather, soil, soil_moisture, vegetation, yield_reference, crop, recommendation, avec source | Conforme | `core/db.py` |
| Jumeau : météo, sol, cultures, croissance, rendements, événements extrêmes ; produit les datasets | Conforme | `core/twin.py`, `core/climate.py`, `core/events.py`, `data/make_datasets.py` |
| 5 modèles ; chacun UNIQUEMENT train.py, predict.py, model.onnx ; indépendants ; colonnes nécessaires seulement | Conforme | métadonnées embarquées dans `model.onnx` ; audit §1 vérifie les fichiers, `SELECT` explicites, aucune lecture croisée |
| Recommendation AI entraînée avec les sorties de référence du jumeau | Conforme | `models/recommendation/train.py` — **conséquence mesurée** : sur la chaîne réelle ses intervalles sont trop étroits (couverture 26 % au lieu de 80 %, audit §3) |
| `main.py` : Climate -> Soil -> Crop -> Yield -> Recommendation, appels successifs | Conforme | `main.py` |
| Mise à jour SQLite + réentraînement, sans changer l'architecture | Conforme | README « Passer aux données réelles » |
| Réponses compréhensibles (format « Culture / Pourquoi / Confiance / Alternative ») | Conforme | `reports/exemples.md` |
| Chaque modèle : prédiction, confiance, incertitude ; détection données jamais vues / hors domaine / forte incertitude ; confiance abaissée automatiquement ; jamais de certitude | Conforme | `_mid`, `_conf`, `_unc`, `ood_flag`, `unseen_features` ; confiance max mesurée 0,95 (plafond 0,85 sans donnée réelle) |
| Surapprentissage, sous-apprentissage, variance, biais, stabilité vérifiés automatiquement + corrections proposées | Conforme | à chaque `train.py` (`advise()`), repris dans l'audit |
| Audit automatique (architecture, données, ML, jumeau) | Conforme | `audit/audit.py`, `audit/audit_twin.py` -> `reports/audit_report.md` |
| Validation scientifique : interne / calibration / externe ; aucune validation externe sans données réelles | Conforme | `audit/validate.py` -> `reports/validation_report.md` : externe et calibration **NON RÉALISÉES** (0 donnée réelle) |
| Livrables (structure, code, doc, scripts train/predict/download/SQLite/audit/validation, rapports) | Conforme | ce dossier |

## Interprétations à connaître (le texte ne les tranche pas)
- Les datasets générés (`ds_main`, `ds_climate`) sont rangés dans la même base SQLite (« une seule base »), en plus des 7 tables listées.
- Code partagé : `core/mlkit.py` (entraînement, conformité, hors-domaine, export ONNX) est utilisé par les 5 `train.py`/`predict.py` plutôt que dupliqué 5 fois (simplicité).
- Export ONNX écrit à la main (le paquet `onnx` était indisponible) ; parité avec scikit-learn mesurée (écart max 6e-4).
- Règle de valeur d'une culture (`core/policy.py`) : choix de conception, aucun prix de marché utilisé.

## Limites mesurées (voir audit)
- Aucun résultat n'est validé sur la réalité : tous les modèles apprennent le jumeau.
- Yield AI sur une région jamais vue : R² 0,45 à 0,54 (contre 0,72 à 0,76 sur régions vues).
- Climate AI : pas de gain sur les pluies extrêmes sur 3 jours ni les jours de chaleur ; gain modeste sur pluie saisonnière (R² 0,15), séquences sèches (0,31).
- Confiance affichée prudente : réussite observée supérieure à la confiance moyenne.
- Déséquilibre des décisions de référence (maïs 59 %, riz pluvial 1 %) issu de la règle de valeur.
