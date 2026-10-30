"""Climate AI : prévoit le climat de la saison (pluie, jours secs d'affilée, pluie extrême, chaleur, besoin d'évaporation) à partir d'informations connues AVANT la saison. Entraîne uniquement ce modèle, sur les colonnes nécessaires de simulation.db."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd
from core.db import connect
from core.mlkit import train_model

NAME = "climate"
FEATURES = ['lat', 'elev', 'win_sin', 'win_cos', 'rain_prev30', 'rain_prev60', 'et0_prev60', 'tmean_prev30', 'rain_norm_season', 'rain_anom_prev60']
TARGETS = ['season_rain_mm', 'dry_spell_days', 'max3_rain_mm', 'heat_days', 'et0_season_mm']
UNITS = {'season_rain_mm': 'mm', 'dry_spell_days': 'jours', 'max3_rain_mm': 'mm', 'heat_days': 'jours', 'et0_season_mm': 'mm'}


def main():
    con = connect()
    cols = FEATURES + TARGETS + ["split", "wseed", "source"]
    df = pd.read_sql_query(f"SELECT {','.join(cols)} FROM ds_climate", con)   # lit uniquement les colonnes nécessaires
    m = train_model(NAME, df, FEATURES, TARGETS, os.path.dirname(os.path.abspath(__file__)), UNITS)
    for t, d in m["diagnostics"].items():
        print(f"{NAME}/{t}: R2 train {d['r2']['tr']:.2f} val {d['r2']['va']:.2f} test {d['r2']['te']:.2f} | couv. test {d['coverage_test']:.0%} | gain vs médiane {d['skill_vs_baseline']:.0%}")
    print("corrections proposées :", *m["advice"], sep="\n  - ")


if __name__ == "__main__":
    main()
