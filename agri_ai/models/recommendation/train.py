"""Recommendation AI : entraînée UNIQUEMENT avec les sorties de RÉFÉRENCE produites par le jumeau numérique
(vérité simulée de la météo, du sol, de la culture et du rendement). En production elle reçoit les sorties des 4 autres modèles."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd
from core.db import connect
from core.mlkit import train_model

NAME = "recommendation"
CROPH = [f"crop_{c}" for c in ["maize", "cassava", "bean", "groundnut", "rice_upland"]]
FEATURES = ["season_rain_mm", "dry_spell_days", "max3_rain_mm", "heat_days",              # référence « Climate AI »
            "erosion_t_ha", "d_soc_pct", "d_ph", "n_leach_kg",                            # référence « Soil AI »
            "days_to_harvest", "water_use_mm", "water_stress", "n_uptake_kg",             # référence « Crop AI »
            "yield_t_ha"] + CROPH                                                        # référence « Yield AI »
TARGETS = ["score_ref"]
UNITS = {"score_ref": "Mkcal/ha/100 j, après pénalités sol/eau"}


def main():
    con = connect()
    sel = ", ".join([f"clim_{c} AS {c}" for c in FEATURES[:4]] + FEATURES[4:] + TARGETS + ["split", "wseed", "source"])   # colonnes nécessaires seulement
    df = pd.read_sql_query(f"SELECT {sel} FROM ds_main", con)
    m = train_model(NAME, df, FEATURES, TARGETS, os.path.dirname(os.path.abspath(__file__)), UNITS)
    d = m["diagnostics"]["score_ref"]
    print(f"recommendation: R2 train {d['r2']['tr']:.2f} val {d['r2']['va']:.2f} test {d['r2']['te']:.2f} | couv. test {d['coverage_test']:.0%}")
    print("corrections proposées :", *m["advice"], sep="\n  - ")


if __name__ == "__main__":
    main()
