"""GPM IMERG (pluie journalière satellite) -> table weather (source='gpm', seule PRECTOTCORR remplie ; sert de RÉFÉRENCE pour contrôler la pluie).
Téléchargement : service « Data Rods for Hydrology » du GES DISC (série temporelle en un point). Variable/version à vérifier à l'usage."""
import re, urllib.request
import pandas as pd
from core.regions import REGIONS
from core.db import put

VAR = "GPM_3IMERGDF_07:precipitationCal"
URL = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"


def url_for(lat, lon, start, end):
    return f"{URL}?variable={VAR}&location=GEOM:POINT({lon},%20{lat})&startDate={start}T00&endDate={end}T23&type=asc2"


def parse(text):
    """Accepte le texte du service (date<espace/tab>valeur) ou un CSV Giovanni (date,valeur)."""
    rows = []
    for l in text.splitlines():
        m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ]?[\d:]*Z?[\s,;]+(-?[\d.]+(?:[eE][-+]?\d+)?)", l.strip())
        if m: rows.append((pd.Timestamp(m.group(1)), float(m.group(2))))
    df = pd.DataFrame(rows, columns=["date", "PRECTOTCORR"])
    return df[(df.PRECTOTCORR >= 0) & (df.PRECTOTCORR < 1000)]       # valeurs de remplissage (-9999) exclues


def load(con, region, start, end, text=None):
    r = REGIONS[region]
    text = text or urllib.request.urlopen(url_for(r.lat, r.lon, start, end), timeout=300).read().decode()
    d = parse(text); d.insert(0, "region", region)
    put(con, "weather", d, "gpm")
