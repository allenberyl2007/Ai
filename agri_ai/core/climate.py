"""Générateur climatique stochastique + ET0 FAO-56 Penman-Monteith.
Sortie = schéma canonique (noms NASA POWER). L'ET0 est RECALCULÉE à partir des variables canoniques
(jamais fournie par le générateur) : le jumeau traite donc de la même façon données simulées et NASA."""
import numpy as np
import pandas as pd
from .regions import Region
from .schema import WEATHER_COLS

SIGMA = 4.903e-9  # MJ K-4 m-2 day-1


def es_kpa(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def extraterrestrial_radiation(lat_deg, doy):
    """Ra (MJ/m2/day), FAO-56 eq. 21."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def _month_interp(values, doy):
    """Interpolation cyclique mensuel -> journalier (points au 15 de chaque mois)."""
    mid = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], float)
    x = np.concatenate([mid - 365, mid, mid + 365])
    y = np.tile(np.asarray(values, float), 3)
    return np.interp(doy, x, y)


def et0_fao56(df: pd.DataFrame, lat: float, elev: float) -> np.ndarray:
    tmax = df["T2M_MAX"].to_numpy(float); tmin = df["T2M_MIN"].to_numpy(float)
    tmean = 0.5 * (tmax + tmin)
    rs = df["ALLSKY_SFC_SW_DWN"].to_numpy(float)
    rh = df["RH2M"].to_numpy(float); u2 = df["WS2M"].to_numpy(float)
    doy = pd.DatetimeIndex(df["date"]).dayofyear.to_numpy()
    P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * P
    es = 0.5 * (es_kpa(tmax) + es_kpa(tmin))
    ea = np.clip(rh / 100.0, 0.0, 1.0) * es
    delta = 4098 * es_kpa(tmean) / (tmean + 237.3) ** 2
    ra = extraterrestrial_radiation(lat, doy)
    rso = (0.75 + 2e-5 * elev) * ra
    rns = 0.77 * rs
    ratio = np.clip(rs / np.maximum(rso, 1e-6), 0.3, 1.0)
    rnl = SIGMA * 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return np.clip(et0, 0.0, 12.0)


def generate_weather(region: Region, start: str, n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy() - 1
    year = dates.year.to_numpy()
    n = n_days

    # --- pluie : multiplicateur annuel + mensuel (gamma), chaîne de Markov d'occurrence ---
    R_clim = np.asarray(region.rain_mm, float)
    ym = {}
    for y in np.unique(year):
        cv = region.rain_cv
        ann = rng.gamma(1 / cv**2, cv**2)
        for m in range(12):
            r = R_clim[m]
            cvm = 0.25 if r > 60 else (0.45 if r > 20 else 0.8)
            mult = rng.gamma(1 / cvm**2, cvm**2)
            ym[(y, m)] = max(0.0, r * (0.6 * ann + 0.4) * mult)  # annuel pondéré 0.6 : conserve la moyenne, ajoute persistance
    rain = np.zeros(n); wet = np.zeros(n, bool)
    rho = 0.25
    f_clim = np.zeros(n)
    prev = False
    for i in range(n):
        R = ym[(year[i], month[i])]
        mu = 6.0 + R / 35.0
        f = min(0.85, R / (30.0 * mu))
        f_clim[i] = f
        p = f * (1 - rho) if not prev else rho + (1 - rho) * f
        w = rng.random() < p
        if w:
            amt = rng.gamma(0.75, mu / 0.75)
            rain[i] = min(amt, 250.0)
            if rain[i] < 0.1: rain[i] = 0.1
        wet[i] = w; prev = w
    wetf = wet.astype(float)

    # --- température ---
    tm = _month_interp(region.tmean, doy)
    dtr = _month_interp(region.dtr, doy)
    a = np.zeros(n); a[0] = rng.normal(0, 1.0)
    for i in range(1, n):
        a[i] = 0.7 * a[i - 1] + rng.normal(0, 0.714)
    dev = wetf - f_clim
    tmax = tm + dtr / 2 + a - 1.8 * dev
    tmin = tm - dtr / 2 + 0.6 * a + 0.6 * dev + rng.normal(0, 0.4, n)
    tmin = np.minimum(tmin, tmax - 1.0)
    t2m = 0.5 * (tmax + tmin)

    # --- rayonnement ---
    Rm = _month_interp(region.rain_mm, doy)
    cloud = Rm / (Rm + 100.0)
    ra = extraterrestrial_radiation(region.lat, doy)
    tau = np.where(wet, np.clip(rng.normal(0.33, 0.09, n), 0.12, 0.60),
                   np.clip(rng.normal(0.68 - 0.14 * cloud, 0.05, n), 0.35, 0.78))
    rs = np.minimum(ra * tau, 0.82 * ra)

    # --- humidité (via point de rosée) ---
    d = np.where(wet, 0.2 + np.abs(rng.normal(0, 0.3, n)), 0.3 + 6.0 * (1 - cloud) ** 2.5 + rng.normal(0, 0.8, n))
    tdew = tmin - np.maximum(d, 0.0)
    ea = es_kpa(tdew)
    es = 0.5 * (es_kpa(tmax) + es_kpa(tmin))
    rh = np.clip(100 * ea / es, 5, 100)

    # --- vent ---
    ws = np.clip(np.exp(rng.normal(np.log(1.2), 0.35, n)), 0.3, 8.0)

    df = pd.DataFrame({"date": dates, "T2M": t2m, "T2M_MAX": tmax, "T2M_MIN": tmin, "PRECTOTCORR": rain,
                       "ALLSKY_SFC_SW_DWN": rs, "RH2M": rh, "WS2M": ws})
    return df[["date"] + WEATHER_COLS]
