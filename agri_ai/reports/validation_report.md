# Rapport de validation scientifique

Trois catégories distinctes. **Aucune validation externe n'est déclarée sans données réelles en base.**


## 1. Validation interne (aucune donnée réelle)

- Jumeau numérique : 96/97 contrôles de cohérence physique/agronomique passés (bilans eau/N/P/K fermés, réponses causales, stabilité, performance). Détail : `reports/twin_checks.md`.
- climate : season_rain_mm R²(test)=0.15, couverture 80 %=86% ; dry_spell_days R²(test)=0.31, couverture 80 %=83% ; max3_rain_mm R²(test)=-0.07, couverture 80 %=82% ; heat_days R²(test)=-0.01, couverture 80 %=98% ; et0_season_mm R²(test)=0.40, couverture 80 %=83%
- soil : erosion_t_ha R²(test)=0.42, couverture 80 %=81% ; d_soc_pct R²(test)=0.83, couverture 80 %=79% ; d_ph R²(test)=0.98, couverture 80 %=81% ; n_leach_kg R²(test)=0.88, couverture 80 %=77%
- crop : days_to_harvest R²(test)=1.00, couverture 80 %=79% ; water_use_mm R²(test)=0.97, couverture 80 %=80% ; water_stress R²(test)=0.75, couverture 80 %=87% ; n_uptake_kg R²(test)=0.89, couverture 80 %=86%
- yield : yield_t_ha R²(test)=0.86, couverture 80 %=82%
- recommendation : score_ref R²(test)=0.99, couverture 80 %=80%

*Portée* : ces chiffres mesurent l'accord des modèles avec **le jumeau numérique** (vérité simulée), pas avec la réalité.


## 2. Calibration

*Ajuste des paramètres avec des données réelles : cela ne prouve rien d'indépendant.*

- **NASA POWER (mensuel, MERRA-2/CERES)** (weather, source `nasa_power_monthly`, 504 lignes) : 
  - pt_0.2987_23.9756 : région simulée la plus proche = kisangani (~138 km) ; 504 mois (1984-2025) ; comparaison indicative (sites différents)
  - température : réel 24.65 vs simulé 25.15 °C (écart +0.51) ; corrélation du cycle mensuel +0.74
  - rayonnement : réel 17.21 vs simulé 17.01 MJ/m²/j (écart -0.20) ; corrélation du cycle mensuel +0.78
  - humidité relative (dérivée de QV2M) : réel 85.19 vs simulé 74.02 pts (écart -11.17) ; corrélation du cycle mensuel +0.57
  - vent à 2 m : réel 0.11 vs simulé 1.28 m/s (écart +1.17) ; corrélation du cycle mensuel -0.48
  - humidité de la zone racinaire : GWETROOT moyen 0.63 vs simulé (sol acrisol_cl, nu) 0.63 ; corrélation du cycle mensuel +0.41
- **NASA POWER (journalier)** (weather, source `nasa_power`, 0 lignes) : NON RÉALISÉE — aucune donnée réelle suffisante en base
- **SoilGrids** (soil, source `soilgrids`, 0 lignes) : NON RÉALISÉE — aucune donnée réelle suffisante en base
- **FAOSTAT** (yield_reference, source `faostat`, 320 lignes) : 
  - bean : FAOSTAT 2010+ 0.54 t/ha (variation 0.54-0.55) ; simulé sans engrais médiane 0.99 (p10-p90 0.27-2.03) ; FAOSTAT dans l'intervalle simulé ; facteur FAOSTAT/simulé 0.55
  - cassava : FAOSTAT 2010+ 8.16 t/ha (variation 8.09-8.42) ; simulé sans engrais médiane 9.63 (p10-p90 1.08-18.23) ; FAOSTAT dans l'intervalle simulé ; facteur FAOSTAT/simulé 0.85
  - groundnut : FAOSTAT 2010+ 0.89 t/ha (variation 0.81-0.99) ; simulé sans engrais médiane 1.06 (p10-p90 0.27-2.13) ; FAOSTAT dans l'intervalle simulé ; facteur FAOSTAT/simulé 0.84
  - maize : FAOSTAT 2010+ 0.77 t/ha (variation 0.77-0.80) ; simulé sans engrais médiane 1.66 (p10-p90 0.38-4.31) ; FAOSTAT dans l'intervalle simulé ; facteur FAOSTAT/simulé 0.47
  - rice_upland : FAOSTAT 2010+ 0.90 t/ha (variation 0.76-1.19) ; simulé sans engrais médiane 1.74 (p10-p90 0.34-3.35) ; FAOSTAT dans l'intervalle simulé ; facteur FAOSTAT/simulé 0.51
  - ATTENTION : 97% des valeurs FAOSTAT 2010+ sont des estimations FAO (drapeau E), quasi constantes d'une année à l'autre : elles donnent un NIVEAU national de référence, aucune information interannuelle ni locale ; un facteur de calage appliqué au jumeau serait une hypothèse

## 3. Validation externe

*Observations indépendantes non utilisées pour ajuster le jumeau.*

- **GPM** (weather, source `gpm`, 0 lignes) : NON RÉALISÉE — aucune donnée réelle suffisante en base
- **SMAP** (soil_moisture, source `smap`, 0 lignes) : NON RÉALISÉE — aucune donnée réelle suffisante en base
- **MODIS NDVI** (vegetation, source `modis`, 0 lignes) : NON RÉALISÉE — aucune donnée réelle suffisante en base
