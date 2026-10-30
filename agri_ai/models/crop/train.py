"""Crop AI : prévoit le comportement de la culture (durée, besoin en eau, stress hydrique, azote absorbé). Entraîne uniquement ce modèle, sur les colonnes nécessaires de simulation.db."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd
from core.db import connect
from core.mlkit import train_model

NAME = "crop"
FEATURES = ['crop_maize', 'crop_cassava', 'crop_bean', 'crop_groundnut', 'crop_rice_upland', 'lat', 'elev', 'win_sin', 'win_cos', 'rain_prev30', 'rain_prev60', 'et0_prev60', 'tmean_prev30', 'rain_norm_season', 'rain_anom_prev60', 'wp', 'fc', 'sat', 'drain', 'cn', 'slope_pct', 'soc_pct', 'ph', 'p_bray', 'k_exch', 'n_kg', 'p_kg', 'k_kg', 'lime_t', 'weeding']
TARGETS = ['days_to_harvest', 'water_use_mm', 'water_stress', 'n_uptake_kg']
UNITS = {'days_to_harvest': 'jours', 'water_use_mm': 'mm', 'water_stress': '0-1', 'n_uptake_kg': 'kg N/ha'}


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
