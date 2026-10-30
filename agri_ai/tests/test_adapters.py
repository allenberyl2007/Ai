"""Tests des adaptateurs sur charges utiles FICTIVES au format documenté (aucun appel réseau)."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from core.db import connect, read_weather, put
from core.schema import WEATHER_COLS
from adapters import power, gpm, smap, modis, soilgrids, faostat

ok = []
def check(name, cond): ok.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)

dates = pd.date_range("2020-01-01", periods=6)
mk = lambda v: {d.strftime("%Y%m%d"): x for d, x in zip(dates, v)}
payload = {"properties": {"parameter": {"T2M": mk([24] * 6), "T2M_MAX": mk([30] * 6), "T2M_MIN": mk([19] * 6),
          "PRECTOTCORR": mk([0, 5, -999, 1, 0, 20]), "ALLSKY_SFC_SW_DWN": mk([15] * 6), "RH2M": mk([70] * 6), "WS2M": mk([1.5] * 6)}}}
df = power.parse(payload)
check("POWER : colonnes canoniques", list(df.columns) == ["date"] + WEATHER_COLS)
check("POWER : -999 -> NaN", df.PRECTOTCORR.isna().sum() == 1)
check("POWER : kWh -> MJ x3.6", abs(power.parse(payload, "kWh").ALLSKY_SFC_SW_DWN.iloc[0] / 15 - 3.6) < 1e-9)
check("POWER : URL contient les 7 paramètres", all(p in power.url_for(-5, 22, "2020-01-01", "2020-01-06") for p in power.PARAMS))

g = gpm.parse("Date&time\tprecipitationCal\n2020-01-01T00:00:00\t3.2\n2020-01-02T00:00:00\t0.0\n2020-01-03T00:00:00\t-9999.9\n")
g2 = gpm.parse("time,mean_precip\n2020-01-01T00:00:00Z,1.5\n")
check("GPM : parse texte du service + CSV, remplissage exclu", len(g) == 2 and g.PRECTOTCORR.iloc[0] == 3.2 and len(g2) == 1)
s = smap.parse("Date,SPL3SMP_E_006_Soil_Moisture_Retrieval_Data_AM_soil_moisture\n2020-01-01,0.31\n2020-01-02,-9999\n")
check("SMAP : remplissage -9999 exclu", len(s) == 1 and abs(s.value.iloc[0] - 0.31) < 1e-9)
m = modis.parse("Date,MOD13Q1_061__250m_16_days_NDVI,MOD13Q1_061__250m_16_days_NDVI_Quality\n2020-01-01,6500,0\n2020-01-17,-3000,0\n")
check("MODIS : échelle 0.0001 et remplissage exclu", len(m) == 1 and abs(m.ndvi.iloc[0] - 0.65) < 1e-9)

sg = {"properties": {"layers": [{"name": n, "depths": [{"label": l, "values": {"mean": v}} for l in ("0-5cm", "5-15cm", "15-30cm")]}
      for n, v in (("phh2o", 52), ("soc", 150), ("clay", 300), ("sand", 450), ("cec", 120))]}}
p = soilgrids.parse(sg)
check("SoilGrids : conversions d'unités (pH 5.2, COS 1.5 %, argile 30 %, sable 45 %, CEC 12)", abs(p["ph"] - 5.2) < 1e-9 and abs(p["soc_pct"] - 1.5) < 1e-9 and p["clay_pct"] == 30 and p["cec"] == 12)
sand = soilgrids.pedotransfer(85, 8, 0.7); clay = soilgrids.pedotransfer(20, 60, 2.5)
check(f"Pedotransfer : sable fc {sand[1]:.2f} < argile fc {clay[1]:.2f} et wp<fc<sat", sand[1] < clay[1] and all(a < b < c for a, b, c in (sand, clay)))
row = soilgrids.to_soil_row("kananga", p, "kananga")
check("SoilGrids : ligne complète pour la table soil", all(row[k] is not None for k in ("wp", "fc", "sat", "drain", "cn", "k_erod", "p_fix", "p_bray", "k_exch")))

fx = pd.DataFrame({"Area": ["Democratic Republic of the Congo"] * 2 + ["France"], "Item": ["Cassava, fresh", "Maize (corn)", "Maize (corn)"],
                   "Element": ["Yield"] * 3, "Year": [2020, 2020, 2020], "Value": [80000, 8000, 90000]})
f = faostat.parse(fx)
check("FAOSTAT : hg/ha -> t/ha et filtre pays", len(f) == 2 and abs(f[f.crop == "cassava"].yield_t_ha.iloc[0] - 8.0) < 1e-9)
fk = faostat.parse(pd.DataFrame({"Area": ["Democratic Republic of the Congo"], "Item": ["Maize (corn)"], "Element": ["Yield"], "Year": [2020], "Unit": ["kg/ha"], "Value": [770.0], "Flag": ["E"]}))
check("FAOSTAT : export du site (kg/ha) -> t/ha et drapeau conservé", abs(fk.yield_t_ha.iloc[0] - 0.77) < 1e-9 and fk.flag.iloc[0] == "E")
mtxt = "-BEGIN HEADER-\nLocation: Latitude  0.2987   Longitude 23.9756 \nElevation from MERRA-2: Average = 444.05 meters\n-END HEADER-\nPARAMETER,YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN\n" + "\n".join(
    f"{k},2000,{','.join([str(v)] * 12)},{v}" for k, v in (("T2M", 24.5), ("ALLSKY_SFC_SW_DWN", 17.0), ("WS2M", 0.1), ("GWETROOT", 0.6), ("QV2M", 17.0)))
kind, mdf, info = power.parse_csv(mtxt)
check("POWER CSV mensuel : détecté, 12 mois, RH dérivée plausible (70-95 %), coordonnées lues", kind == "monthly" and len(mdf) == 12 and 70 < mdf.RH2M.iloc[0] < 95 and abs(info["lat"] - 0.2987) < 1e-6 and info["elev"] == 444.05)
dtxt = "-BEGIN HEADER-\n-END HEADER-\nYEAR,MO,DY," + ",".join(power.PARAMS) + "\n2020,1,1,24,30,19,5,15,70,1.5\n2020,1,2,24,30,19,-999,15,70,1.5\n"
kd, ddf, _ = power.parse_csv(dtxt)
check("POWER CSV journalier : 7 paramètres, -999 -> NaN", kd == "daily" and len(ddf) == 2 and ddf.PRECTOTCORR.isna().sum() == 1)
try:
    power.parse_csv("-END HEADER-\nYEAR,MO,DY,T2M\n2020,1,1,24\n"); check("POWER CSV journalier incomplet refusé avec message", False)
except ValueError:
    check("POWER CSV journalier incomplet refusé avec message", True)

with tempfile.TemporaryDirectory() as t:
    con = connect(os.path.join(t, "x.db"))
    d = df.dropna().copy(); d.insert(0, "region", "kananga"); put(con, "weather", d, "nasa_power")
    d2 = d.copy(); d2["PRECTOTCORR"] += 100; put(con, "weather", d2, "sim")
    r = read_weather(con, "kananga", "2020-01-01", "2020-01-06", ("nasa_power", "sim"))
    check("DB : priorité réel > simulé, jour par jour", (r.source == "nasa_power").all() and len(r) == 5)
    tabs = {x[0] for x in con.execute("select name from sqlite_master where type='table'")}
    check("DB : 7 tables requises présentes", {"weather", "soil", "soil_moisture", "vegetation", "yield_reference", "crop", "recommendation"} <= tabs)
    cols = {tb: [c[1] for c in con.execute(f"pragma table_info({tb})")] for tb in tabs}
    check("DB : chaque table garde sa source", all("source" in cols[tb] for tb in ("weather", "soil", "soil_moisture", "vegetation", "yield_reference", "crop", "recommendation")))
print(f"\n{sum(1 for _, c in ok if c)}/{len(ok)} tests adaptateurs OK")
sys.exit(0 if all(c for _, c in ok) else 1)
