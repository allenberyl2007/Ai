"""Soil AI : prévoit l'effet d'une culture sur le sol (érosion, carbone, acidité, pertes d'azote). Entraîne uniquement ce modèle, sur les colonnes nécessaires de simulation.db."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd
from core.db import connect
from core.mlkit import train_model

NAME = "soil"
FEATURES = ['soc_pct', 'ph', 'p_bray', 'k_exch', 'cec', 'wp', 'fc', 'sat', 'drain', 'cn', 'k_erod', 'slope_pct', 'n_kg', 'p_kg', 'k_kg', 'lime_t', 'weeding', 'residue_return', 'crop_maize', 'crop_cassava', 'crop_bean', 'crop_groundnut', 'crop_rice_upland', 'rain_norm_season', 'rain_anom_prev60', 'rain_prev60']
TARGETS = ['erosion_t_ha', 'd_soc_pct', 'd_ph', 'n_leach_kg']
UNITS = {'erosion_t_ha': 't/ha', 'd_soc_pct': 'points de %', 'd_ph': 'unités pH', 'n_leach_kg': 'kg N/ha'}


def main():
    con = connect()
    cols = FEATURES + TARGETS + ["split", "wseed", "source"]
    df = pd.read_sql_query(f"SELECT {','.join(cols)} FROM ds_main", con)   # lit uniquement les colonnes nécessaires
    m = train_model(NAME, df, FEATURES, TARGETS, os.path.dirname(os.path.abspath(__file__)), UNITS)
    for t, d in m["diagnostics"].items():
        print(f"{NAME}/{t}: R2 train {d['r2']['tr']:.2f} val {d['r2']['va']:.2f} test {d['r2']['te']:.2f} | couv. test {d['coverage_test']:.0%} | gain vs médiane {d['skill_vs_baseline']:.0%}")
    print("corrections proposées :", *m["advice"], sep="\n  - ")


if __name__ == "__main__":
    main()
