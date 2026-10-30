"""NASA POWER (météo journalière) -> table weather (source='nasa_power'). Format : properties.parameter.<VAR>.<AAAAMMJJ>, remplissage -999."""
import io, re
import numpy as np
import pandas as pd
from core.schema import WEATHER_COLS, clean_weather, repair_weather, validate_weather
from core.regions import REGIONS
from core.db import put
from . import http_json

PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M"]
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def url_for(lat, lon, start, end, community="AG"):
    return (f"{URL}?parameters={','.join(PARAMS)}&community={community}&longitude={lon}&latitude={lat}"
            f"&start={start.replace('-', '')}&end={end.replace('-', '')}&format=JSON")


def parse(payload, sw_unit="MJ"):
    """sw_unit 'MJ' (communauté AG) ou 'kWh' (RE) -> converti en MJ/m2/jour."""
    par = payload["properties"]["parameter"]; idx = sorted(par[PARAMS[0]])
    df = pd.DataFrame({"date": pd.to_datetime(idx, format="%Y%m%d")})
    for v in PARAMS: df[v] = [par[v][k] for k in idx]
    df = clean_weather(df)
    if sw_unit.lower().startswith("kwh"): df["ALLSKY_SFC_SW_DWN"] *= 3.6
    return df[["date"] + WEATHER_COLS]


def load(con, region, start, end):
    r = REGIONS[region]
    payload = http_json(url_for(r.lat, r.lon, start, end))
    units = str(payload.get("parameters", {}).get("ALLSKY_SFC_SW_DWN", {}).get("units", ""))
    df, rep = repair_weather(parse(payload, "kWh" if "kW-hr" in units else "MJ"), 5)
    qc = validate_weather(df.dropna())
    d = df.dropna().copy(); d.insert(0, "region", region)
    put(con, "weather", d, "nasa_power")
    return qc


def _rh_from_q(q_gkg, t_c, elev):
    """RH approchée (%) depuis l'humidité spécifique (g/kg), T moyenne et altitude : e = qP/(0.622+0.378q), RH = e/es(Tmoy)."""
    P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26; q = q_gkg / 1000.0
    return np.clip(100 * (q * P / (0.622 + 0.378 * q)) / (0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))), 5, 100)


def parse_csv(text):
    """CSV téléchargé depuis le site NASA POWER. Retourne (kind, DataFrame, info) ; kind = 'monthly' ou 'daily'."""
    header, body = text.split("-END HEADER-", 1) if "-END HEADER-" in text else ("", text)
    m = re.search(r"Latitude\s+(-?[\d.]+)\s+Longitude\s+(-?[\d.]+)", header); e = re.search(r"=\s*([\d.]+)\s*meters", header)
    info = dict(lat=float(m.group(1)) if m else None, lon=float(m.group(2)) if m else None, elev=float(e.group(1)) if e else 0.0)
    df = pd.read_csv(io.StringIO(body.strip())).replace(-999, np.nan)
    if "PARAMETER" in df.columns:                       # mensuel : une ligne par paramètre et par année
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        long = df.melt(id_vars=["PARAMETER", "YEAR"], value_vars=months, var_name="m", value_name="v")
        long["date"] = pd.to_datetime(dict(year=long.YEAR, month=long.m.map({k: i + 1 for i, k in enumerate(months)}), day=1))
        out = long.pivot_table(index="date", columns="PARAMETER", values="v").reset_index()
        if "QV2M" in out and "T2M" in out: out["RH2M"] = _rh_from_q(out["QV2M"], out["T2M"], info["elev"])
        return "monthly", out.drop(columns=[c for c in ("QV2M",) if c in out]), info
    if "DOY" in df.columns: df["date"] = pd.to_datetime(df.YEAR.astype(str)) + pd.to_timedelta(df.DOY - 1, unit="D")
    else: df["date"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY))
    miss = [c for c in PARAMS if c not in df.columns]
    if miss: raise ValueError(f"paramètres manquants dans le CSV journalier : {miss} — télécharger les 7 : {PARAMS}")
    return "daily", clean_weather(df[["date"] + PARAMS]), info


def load_csv(con, region, path):
    """Mensuel -> weather (source 'nasa_power_monthly', jamais lu par le jumeau) + soil_moisture (GWETROOT). Journalier -> weather (source 'nasa_power')."""
    kind, df, info = parse_csv(open(path, encoding="utf-8", errors="ignore").read())
    d = df.copy(); d.insert(0, "region", region)
    if kind == "daily":
        d, _ = repair_weather(d.drop(columns="region"), 5); d = d.dropna().copy(); d.insert(0, "region", region); put(con, "weather", d, "nasa_power")
    else:
        if "GWETROOT" in d:
            put(con, "soil_moisture", d[["region", "date", "GWETROOT"]].rename(columns={"GWETROOT": "value"}).dropna().assign(kind="gwetroot_monthly"), "nasa_power_monthly")
        put(con, "weather", d.drop(columns=[c for c in ("GWETROOT",) if c in d]), "nasa_power_monthly")
    return dict(kind=kind, rows=len(df), **info)
