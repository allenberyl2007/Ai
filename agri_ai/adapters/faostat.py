"""FAOSTAT (production agricole, fichier bulk normalisé) -> yield_reference (t/ha). Niveau NATIONAL et ANNUEL :
sert à caler les niveaux de rendement, pas à valider des parcelles."""
import io, zipfile, urllib.request
import pandas as pd
from core.db import put

URL = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
ITEMS = {"Maize (corn)": "maize", "Cassava, fresh": "cassava", "Beans, dry": "bean",
         "Groundnuts, excluding shelled": "groundnut", "Rice": "rice_upland"}   # Rice : paddy national (pluvial + irrigué)


UNIT_TO_T_HA = {"kg/ha": 1e-3, "hg/ha": 1e-4, "t/ha": 1.0}


def parse(df, area="Democratic Republic of the Congo"):
    """Fichier bulk (hg/ha) ou export du site FAOSTAT (kg/ha) : l'unité est lue dans la colonne Unit. Le drapeau de qualité est conservé
    (E = valeur estimée par la FAO, A = officielle...) : indispensable pour savoir ce que ces rendements valent."""
    d = df[(df["Area"] == area) & (df["Element"] == "Yield") & (df["Item"].isin(ITEMS))].copy()
    unit = d["Unit"] if "Unit" in d else pd.Series("hg/ha", index=d.index)
    d["yield_t_ha"] = pd.to_numeric(d["Value"], errors="coerce") * unit.map(UNIT_TO_T_HA)
    d["crop"] = d["Item"].map(ITEMS); d["country"] = "COD"; d["year"] = d["Year"].astype(int)
    d["flag"] = d["Flag"] if "Flag" in d else None
    return d[["country", "crop", "year", "yield_t_ha", "flag"]].dropna(subset=["yield_t_ha"])


def load_csv(con, path, area="Democratic Republic of the Congo"):
    """Importe un export FAOSTAT téléchargé à la main (CSV)."""
    df = pd.read_csv(path, encoding="utf-8-sig"); put(con, "yield_reference", parse(df, area), "faostat")
    return len(df)


def download():
    raw = urllib.request.urlopen(URL, timeout=600).read()
    z = zipfile.ZipFile(io.BytesIO(raw)); name = next(n for n in z.namelist() if n.endswith(".csv"))
    return pd.read_csv(z.open(name), encoding="latin-1", usecols=["Area", "Item", "Element", "Year", "Value"])


def load(con, df=None):
    put(con, "yield_reference", parse(download() if df is None else df), "faostat")
