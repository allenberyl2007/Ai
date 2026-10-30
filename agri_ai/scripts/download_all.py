"""Télécharge les sources réelles dans simulation.db (nécessite Internet ; SMAP/MODIS : EARTHDATA_USER/EARTHDATA_PASS).
Chaque source est indépendante : un échec n'arrête pas les autres. Usage : python scripts/download_all.py kananga 2015-01-01 2024-12-31"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.db import connect
from adapters import power, soilgrids, faostat, smap, modis, gpm

region, start, end = sys.argv[1:4]
con = connect()
for name, fn in (("NASA POWER", lambda: power.load(con, region, start, end)),
                 ("SoilGrids", lambda: soilgrids.load(con, region)),
                 ("FAOSTAT", lambda: faostat.load(con)),
                 ("SMAP", lambda: smap.load(con, region, start, end)),
                 ("MODIS NDVI", lambda: modis.load(con, region, start, end)),
                 ("GPM", lambda: gpm.load(con, region, start, end))):
    try:
        print(name, "OK", fn() or "")
    except Exception as e:
        print(name, "ÉCHEC :", type(e).__name__, str(e)[:150])
