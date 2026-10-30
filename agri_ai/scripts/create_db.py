"""Crée data/simulation.db : tables vides + archétypes de sols + paramètres des cultures (sources tracées)."""
import sys, os, json, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from core.db import connect, put
from core.regions import SOILS
from core.crops import CROPS

con = connect()
rows = [dict(site=k, soc_pct=s.soc, ph=s.ph, p_bray=s.p_bray, k_exch=s.k_exch, cec=s.cec, wp=s.wp, fc=s.fc, sat=s.sat,
             drain=s.drain, cn=s.cn, k_erod=s.k_erod, p_fix=s.p_fix, sand_pct=None, clay_pct=None,
             notes="archétype : ordres de grandeur de la littérature, non mesuré") for k, s in SOILS.items()]
put(con, "soil", pd.DataFrame(rows), "archetype")
put(con, "crop", pd.DataFrame([dict(name=k, params=json.dumps(dataclasses.asdict(c))) for k, c in CROPS.items()]), "literature")
print("base créée :", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "simulation.db")))
