"""MODIS NDVI (MOD13Q1, 16 jours, 250 m) -> vegetation. Donnée d'observation (contrôle de la couverture végétale simulée)."""
import io
import pandas as pd
from core.regions import REGIONS
from core.db import put
from .smap import point_csv

PRODUCT, LAYER = "MOD13Q1.061", "_250m_16_days_NDVI"


def parse(text):
    df = pd.read_csv(io.StringIO(text))
    col = next(c for c in df.columns if "ndvi" in c.lower() and "quality" not in c.lower())
    v = pd.to_numeric(df[col], errors="coerce")
    if v.abs().max() > 2: v = v * 0.0001            # valeurs brutes MODIS (facteur d'échelle 0.0001)
    out = pd.DataFrame({"date": pd.to_datetime(df[next(c for c in df.columns if c.lower() == "date")]), "ndvi": v})
    return out[(out.ndvi > -0.2) & (out.ndvi <= 1.0)].dropna()


def load(con, region, start, end, csv_text=None):
    r = REGIONS[region]
    text = csv_text or point_csv(r.lat, r.lon, start, end, PRODUCT, LAYER, "ndvi")
    d = parse(text); d.insert(0, "region", region)
    put(con, "vegetation", d, "modis")
