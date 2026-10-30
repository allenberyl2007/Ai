"""Événements extrêmes et scénarios : surcouches appliquées aux entrées météo (ou à l'état initial du sol,
ou aux dommages biotiques). Les données de base restent intactes (traçabilité)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .climate import es_kpa

KINDS = ("drought", "dry_spell", "flood", "heatwave", "pest", "disease", "fertility_loss")


@dataclass(frozen=True)
class Event:
    kind: str
    day: int = 0            # indice de jour dans la simulation (0 = 1er jour)
    duration: int = 10
    intensity: float = 0.5  # drought/pest/disease/fertility_loss: fraction 0-1 ; flood: mm/jour ; heatwave: +degC

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"événement inconnu: {self.kind}")


def apply_weather_events(df: pd.DataFrame, events) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    for e in events:
        a, b = max(0, e.day), min(n, e.day + e.duration)
        if a >= b:
            continue
        sl = slice(a, b)
        if e.kind == "drought":
            df.loc[df.index[sl], "PRECTOTCORR"] *= (1.0 - e.intensity)
        elif e.kind == "dry_spell":
            df.loc[df.index[sl], "PRECTOTCORR"] = 0.0
        elif e.kind == "flood":
            df.loc[df.index[sl], "PRECTOTCORR"] += e.intensity
            df.loc[df.index[sl], "ALLSKY_SFC_SW_DWN"] *= 0.7
        elif e.kind == "heatwave":
            old_es = 0.5 * (es_kpa(df["T2M_MAX"].iloc[sl]) + es_kpa(df["T2M_MIN"].iloc[sl]))
            df.iloc[sl, df.columns.get_loc("T2M_MAX")] += e.intensity
            df.iloc[sl, df.columns.get_loc("T2M_MIN")] += 0.6 * e.intensity
            df.iloc[sl, df.columns.get_loc("T2M")] = 0.5 * (df["T2M_MAX"].iloc[sl] + df["T2M_MIN"].iloc[sl])
            new_es = 0.5 * (es_kpa(df["T2M_MAX"].iloc[sl]) + es_kpa(df["T2M_MIN"].iloc[sl]))
            df.iloc[sl, df.columns.get_loc("RH2M")] = np.clip(df["RH2M"].iloc[sl] * old_es / new_es, 5, 100)
            df.iloc[sl, df.columns.get_loc("ALLSKY_SFC_SW_DWN")] *= 1.05
    return df


def biotic_arrays(events, n):
    """pest[i] : fraction de perte de croissance/jour ; disease_mult[i] : multiplicateur d'infection."""
    pest = np.zeros(n); dis = np.ones(n)
    for e in events:
        a, b = max(0, e.day), min(n, e.day + e.duration)
        if a >= b:
            continue
        if e.kind == "pest":
            pest[a:b] = np.maximum(pest[a:b], 1 - (1 - e.intensity) ** (1.0 / max(1, e.duration)))   # couverture restante = 1-intensité
        elif e.kind == "disease":
            dis[a:b] *= (1.0 + 4.0 * e.intensity)
    return pest, dis


def soil_fertility_loss(events):
    f = 0.0
    for e in events:
        if e.kind == "fertility_loss":
            f = max(f, e.intensity)
    return f
