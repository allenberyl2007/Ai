"""Schéma canonique des variables. UNE seule définition (noms NASA POWER, unités SI-agro).
Toute source (simulation, NASA POWER, station, CHIRPS...) est convertie vers ce schéma
par un adaptateur; le jumeau ne lit JAMAIS autre chose que ce schéma."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Var:
    name: str
    unit: str
    lo: float
    hi: float
    desc: str


WEATHER_VARS = [
    Var("T2M", "degC", -5.0, 45.0, "Température moyenne journalière à 2 m"),
    Var("T2M_MAX", "degC", -5.0, 50.0, "Température max journalière à 2 m"),
    Var("T2M_MIN", "degC", -10.0, 40.0, "Température min journalière à 2 m"),
    Var("PRECTOTCORR", "mm/day", 0.0, 500.0, "Précipitation corrigée"),
    Var("ALLSKY_SFC_SW_DWN", "MJ/m2/day", 0.0, 35.0, "Rayonnement solaire descendant (ciel réel)"),
    Var("RH2M", "%", 5.0, 100.0, "Humidité relative à 2 m"),
    Var("WS2M", "m/s", 0.0, 25.0, "Vitesse du vent à 2 m"),
]
WEATHER_COLS = [v.name for v in WEATHER_VARS]

# Variables d'observation utilisées pour CALIBRER l'état du sol simulé (pas des entrées du jumeau)
OBS_VARS = [
    Var("GWETROOT", "fraction", 0.0, 1.0, "NASA POWER : humidité de la zone racinaire (0-1)"),
    Var("SM_SURF", "m3/m3", 0.0, 0.7, "SMAP : humidité volumique de surface"),
    Var("NDVI", "-", -0.1, 1.0, "MODIS/Sentinel : indice de végétation"),
]

NASA_POWER_FILL = -999.0


def validate_weather(df: pd.DataFrame) -> dict:
    """Contrôle qualité : colonnes, plages, cohérence physique. Renvoie un rapport (jamais d'exception
    silencieuse : le rapport est stocké avec les données)."""
    rep = {"n": len(df), "missing_cols": [c for c in WEATHER_COLS if c not in df.columns]}
    if rep["missing_cols"]:
        rep["ok"] = False
        return rep
    out_of_range = {}
    for v in WEATHER_VARS:
        s = df[v.name]
        out_of_range[v.name] = int(((s < v.lo) | (s > v.hi)).sum())
    rep["out_of_range"] = out_of_range
    rep["n_nan"] = int(df[WEATHER_COLS].isna().sum().sum())
    rep["tmax_lt_tmin"] = int((df["T2M_MAX"] < df["T2M_MIN"]).sum())
    rep["tmean_outside_minmax"] = int(((df["T2M"] > df["T2M_MAX"] + 1e-6) | (df["T2M"] < df["T2M_MIN"] - 1e-6)).sum())
    rep["ok"] = (sum(out_of_range.values()) + rep["n_nan"] + rep["tmax_lt_tmin"]) == 0
    return rep


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace valeurs de remplissage NASA par NaN et force la cohérence Tmin<=T2M<=Tmax."""
    df = df.copy()
    df[WEATHER_COLS] = df[WEATHER_COLS].where(df[WEATHER_COLS] != NASA_POWER_FILL, np.nan)
    return df


def repair_weather(df: pd.DataFrame, max_gap: int = 5):
    """Interpolation linéaire des trous courts (<= max_gap jours). Les trous plus longs restent NaN
    (=> le jumeau refuse de tourner : on ne fabrique pas de données). Retourne (df, rapport)."""
    df = clean_weather(df)
    rep = {}
    for c in WEATHER_COLS:
        before = int(df[c].isna().sum())
        df[c] = df[c].interpolate(limit=max_gap, limit_area="inside")
        rep[c] = {"filled": before - int(df[c].isna().sum()), "remaining": int(df[c].isna().sum())}
    df["T2M_MAX"] = np.maximum(df["T2M_MAX"], df["T2M_MIN"])
    df["PRECTOTCORR"] = df["PRECTOTCORR"].clip(lower=0)
    return df, rep
