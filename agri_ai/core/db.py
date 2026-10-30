"""Une seule base SQLite (data/simulation.db). Chaque table garde la colonne `source`."""
import sqlite3, os
import pandas as pd
from .schema import WEATHER_COLS

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "simulation.db")

DDL = [
    f"CREATE TABLE IF NOT EXISTS weather (region TEXT, date TEXT, {', '.join(c + ' REAL' for c in WEATHER_COLS)}, source TEXT, PRIMARY KEY(region,date,source))",
    "CREATE TABLE IF NOT EXISTS soil (site TEXT, source TEXT, soc_pct REAL, ph REAL, p_bray REAL, k_exch REAL, cec REAL, wp REAL, fc REAL, sat REAL, drain REAL, cn REAL, k_erod REAL, p_fix REAL, sand_pct REAL, clay_pct REAL, notes TEXT, PRIMARY KEY(site,source))",
    "CREATE TABLE IF NOT EXISTS soil_moisture (region TEXT, date TEXT, value REAL, kind TEXT, source TEXT, PRIMARY KEY(region,date,kind,source))",
    "CREATE TABLE IF NOT EXISTS vegetation (region TEXT, date TEXT, ndvi REAL, source TEXT, PRIMARY KEY(region,date,source))",
    "CREATE TABLE IF NOT EXISTS yield_reference (country TEXT, crop TEXT, year INTEGER, yield_t_ha REAL, flag TEXT, source TEXT, PRIMARY KEY(country,crop,year,source))",
    "CREATE TABLE IF NOT EXISTS crop (name TEXT, params TEXT, source TEXT, PRIMARY KEY(name,source))",
    "CREATE TABLE IF NOT EXISTS recommendation (ts TEXT, region TEXT, crop TEXT, score REAL, confidence REAL, payload TEXT, source TEXT)",
]


def connect(path=None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path)
    for q in DDL:
        con.execute(q)
    if "notes" not in [r[1] for r in con.execute("PRAGMA table_info(soil)")]:      # migration d'anciennes bases
        con.execute("ALTER TABLE soil ADD COLUMN notes TEXT")
    if "flag" not in [r[1] for r in con.execute("PRAGMA table_info(yield_reference)")]:
        con.execute("ALTER TABLE yield_reference ADD COLUMN flag TEXT")
    return con


def put(con, table, df, source, replace_key=None):
    """Ajoute des lignes avec leur source (INSERT OR REPLACE sur la clé primaire de la table)."""
    d = df.copy(); d["source"] = source
    if "date" in d: d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    cols = list(d.columns)
    con.executemany(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    d.itertuples(index=False, name=None))
    con.commit()


def read_weather(con, region, start, end, sources=("nasa_power", "sim")):
    """Lit la météo en respectant l'ordre de priorité des sources (réel avant simulé, jour par jour)."""
    q = ",".join("?" * len(sources))
    df = pd.read_sql_query(f"SELECT * FROM weather WHERE region=? AND date>=? AND date<=? AND source IN ({q})",
                           con, params=(region, start, end, *sources))
    if df.empty:
        return df
    rank = {s: i for i, s in enumerate(sources)}
    df["_r"] = df["source"].map(rank)
    df = df.sort_values(["date", "_r"]).drop_duplicates("date").drop(columns="_r")
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)
