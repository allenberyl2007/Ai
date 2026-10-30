"""Variables d'entrée « connues AVANT la saison » et cibles climatiques « futures ».
Règle anti-fuite : pre_season_features() n'utilise QUE des jours strictement antérieurs à l'indice de semis."""
import numpy as np
import pandas as pd
from .regions import REGIONS
from .climate import et0_fao56

DAYS_IN = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
HORIZON = 120   # jours de saison suivis par l'IA climat


def _normal_daily(values, doys):
    """Normale journalière : valeur mensuelle répartie sur les jours du mois."""
    m = np.minimum(11, ((np.asarray(doys) - 1) / 30.4375).astype(int))
    return np.asarray(values, float)[m] / DAYS_IN[m]


def pre_season_features(w: pd.DataFrame, region: str, idx: int, et0=None) -> dict:
    """w : météo canonique ; idx : indice du jour de début de fenêtre de semis. Utilise w[:idx] uniquement
    (+ normales climatiques de la région, connues d'avance)."""
    r = REGIONS[region]
    doy = pd.DatetimeIndex(w["date"]).dayofyear.to_numpy()
    if et0 is None:
        et0 = et0_fao56(w.iloc[:idx], r.lat, r.elev)
    rain = w["PRECTOTCORR"].to_numpy(float)
    tm = w["T2M"].to_numpy(float)
    fut = (doy[idx] + np.arange(HORIZON) - 1) % 365 + 1
    pre60 = doy[idx - 60:idx]
    rn_s = _normal_daily(r.rain_mm, fut).sum()
    rn_p60 = _normal_daily(r.rain_mm, pre60).sum()
    ang = 2 * np.pi * doy[idx] / 365.0
    return dict(
        lat=r.lat, elev=r.elev, win_sin=float(np.sin(ang)), win_cos=float(np.cos(ang)),
        rain_prev30=float(rain[idx - 30:idx].sum()), rain_prev60=float(rain[idx - 60:idx].sum()),
        et0_prev60=float(np.asarray(et0)[idx - 60:idx].sum()), tmean_prev30=float(tm[idx - 30:idx].mean()),
        rain_norm_season=float(rn_s),
        rain_anom_prev60=float(rain[idx - 60:idx].sum() / (rn_p60 + 10.0)),
    )


def season_targets(w: pd.DataFrame, region: str, idx: int) -> dict:
    """Cibles climatiques FUTURES (jours idx .. idx+HORIZON) — jamais utilisées comme entrées."""
    r = REGIONS[region]
    sl = slice(idx, idx + HORIZON)
    rain = w["PRECTOTCORR"].to_numpy(float)[sl]
    tmax = w["T2M_MAX"].to_numpy(float)[sl]
    et0 = et0_fao56(w.iloc[sl], r.lat, r.elev)
    run = best = 0
    for d in (rain < 1.0):
        run = run + 1 if d else 0
        best = max(best, run)
    return dict(season_rain_mm=float(rain.sum()), dry_spell_days=float(best),
                max3_rain_mm=float(np.convolve(rain, np.ones(3), mode="valid").max()),
                heat_days=float((tmax >= 35.0).sum()), et0_season_mm=float(et0.sum()))
