"""ISRIC SoilGrids 2.0 (REST) -> soil. Valeurs 0-30 cm (moyenne pondérée par l'épaisseur des 3 couches 0-5, 5-15, 15-30 cm).
Unités SoilGrids : phh2o pH*10 ; soc dg/kg ; clay/sand g/kg ; cec mmol(c)/kg. SoilGrids ne contient PAS P ni K :
ils sont repris de l'archétype régional (note dans la colonne `notes`) -> confiance réduite."""
import numpy as np
import pandas as pd
from core.regions import REGIONS, SOILS
from core.db import put
from . import http_json

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS = ["phh2o", "soc", "clay", "sand", "cec"]
W = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}


def parse(payload):
    out = {}
    for layer in payload["properties"]["layers"]:
        num = den = 0.0
        for dep in layer["depths"]:
            v = dep["values"].get("mean")
            if v is None: continue
            w = W[dep["label"]]; num += v * w; den += w
        out[layer["name"]] = num / den if den else np.nan
    return dict(ph=out["phh2o"] / 10.0, soc_pct=out["soc"] / 100.0, clay_pct=out["clay"] / 10.0, sand_pct=out["sand"] / 10.0, cec=out["cec"] / 10.0)


def pedotransfer(sand_pct, clay_pct, soc_pct):
    """Saxton & Rawls (2006) : points de flétrissement (1500 kPa), capacité au champ (33 kPa), saturation (m3/m3)."""
    S, C, OM = sand_pct / 100.0, clay_pct / 100.0, soc_pct * 1.724
    t1500 = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    wp = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    fc = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    ts = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    sat = fc + (ts + (0.636 * ts - 0.107)) - 0.097 * S + 0.043
    wp = float(np.clip(wp, 0.03, 0.30)); fc = float(np.clip(fc, wp + 0.04, 0.48)); sat = float(np.clip(sat, fc + 0.03, 0.60))
    return wp, fc, sat


def to_soil_row(site, p, region):
    wp, fc, sat = pedotransfer(p["sand_pct"], p["clay_pct"], p["soc_pct"])
    arche = [SOILS[s] for s in REGIONS[region].soils]
    return dict(site=site, soc_pct=p["soc_pct"], ph=p["ph"], p_bray=float(np.median([a.p_bray for a in arche])),
                k_exch=float(np.median([a.k_exch for a in arche])), cec=p["cec"], wp=wp, fc=fc, sat=sat,
                drain=float(np.clip(0.25 + 0.65 * p["sand_pct"] / 100.0, 0.2, 0.95)),                 # heuristique texture -> drainage
                cn=float(np.clip(92 - 0.28 * p["sand_pct"] + 0.1 * p["clay_pct"], 60, 90)),           # heuristique texture -> ruissellement
                k_erod=float(np.clip(0.005 + 0.0003 * (100 - p["clay_pct"]) * (1 - p["soc_pct"] / 10), 0.005, 0.04)),
                p_fix=float(np.clip(0.1 + 0.012 * p["clay_pct"], 0.1, 0.9)), sand_pct=p["sand_pct"], clay_pct=p["clay_pct"],
                notes="P et K = médiane des archétypes de la région (absents de SoilGrids)")


def load(con, region, lat=None, lon=None):
    r = REGIONS[region]; lat = r.lat if lat is None else lat; lon = r.lon if lon is None else lon
    q = f"{URL}?lon={lon}&lat={lat}&" + "&".join(f"property={p}" for p in PROPS) + "&" + "&".join(f"depth={d}" for d in W) + "&value=mean"
    row = to_soil_row(region, parse(http_json(q)), region)
    put(con, "soil", pd.DataFrame([row]), "soilgrids")
