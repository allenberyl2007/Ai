"""SMAP (humidité du sol de surface, m3/m3) -> soil_moisture (kind='smap_surface'). Donnée d'OBSERVATION : sert à calibrer/valider, pas d'entrée du jumeau.
Produit/couche AppEEARS à vérifier à l'usage (version du produit susceptible de changer)."""
import io
import pandas as pd
from core.regions import REGIONS
from core.db import put
import os, time, base64, urllib.request
from . import http_json

PRODUCT, LAYER = "SPL3SMP_E.006", "Soil_Moisture_Retrieval_Data_AM_soil_moisture"
API = "https://appeears.earthdatacloud.nasa.gov/api"


def point_csv(lat, lon, start, end, product, layer, name="agri"):
    user, pw = os.environ["EARTHDATA_USER"], os.environ["EARTHDATA_PASS"]
    tok = base64.b64encode(f"{user}:{pw}".encode()).decode()
    token = http_json(f"{API}/login", data={}, headers={"Authorization": f"Basic {tok}"})["token"]
    hdr = {"Authorization": f"Bearer {token}"}
    task = http_json(f"{API}/task", headers=hdr, data={"task_type": "point", "task_name": name, "params": {
        "dates": [{"startDate": pd_date(start), "endDate": pd_date(end)}], "layers": [{"product": product, "layer": layer}],
        "coordinates": [{"id": "p1", "latitude": lat, "longitude": lon, "category": "agri"}]}})
    tid = task["task_id"]
    while http_json(f"{API}/status/{tid}", headers=hdr)[0].get("status") != "done":
        time.sleep(20)
    files = http_json(f"{API}/bundle/{tid}", headers=hdr)["files"]
    fid = next(f["file_id"] for f in files if f["file_name"].endswith(".csv") and "Statistics" not in f["file_name"])
    req = urllib.request.Request(f"{API}/bundle/{tid}/{fid}", headers=hdr)
    return urllib.request.urlopen(req, timeout=300).read().decode()



def pd_date(s):   # AppEEARS veut MM-JJ-AAAA
    y, m, d = s.split("-"); return f"{m}-{d}-{y}"



def parse(text):
    df = pd.read_csv(io.StringIO(text))
    col = next(c for c in df.columns if "soil_moisture" in c.lower())
    out = pd.DataFrame({"date": pd.to_datetime(df[next(c for c in df.columns if c.lower() == "date")]),
                        "value": pd.to_numeric(df[col], errors="coerce")})
    return out[(out.value > 0) & (out.value < 0.8)].dropna()     # -9999 = remplissage ; plage physique m3/m3


def load(con, region, start, end, csv_text=None):
    r = REGIONS[region]
    text = csv_text or point_csv(r.lat, r.lon, start, end, PRODUCT, LAYER, "smap")
    d = parse(text); d.insert(0, "region", region); d["kind"] = "smap_surface"
    put(con, "soil_moisture", d, "smap")
