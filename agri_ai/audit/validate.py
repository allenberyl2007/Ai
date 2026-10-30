"""Validation scientifique : compare le jumeau numérique aux références réelles quand elles sont présentes dans simulation.db.
Trois catégories STRICTEMENT distinctes :
  - Validation interne   : cohérence du système avec lui-même (bilans, jumeau vs modèles). Aucune donnée réelle.
  - Calibration          : ajustement de paramètres avec des données réelles (POWER, SoilGrids, FAOSTAT). Ne prouve rien d'externe.
  - Validation externe   : comparaison à des observations indépendantes (GPM, SMAP, MODIS). SANS données réelles : « NON RÉALISÉE ».
Usage : python audit/validate.py [chemin_db]   ->  reports/validation_report.md"""
import sys, os, json, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from core.db import connect
from core.regions import REGIONS
from core.climate import generate_weather, et0_fao56
from core.twin import DigitalTwin, Site, Management
from core.mlkit import read_meta

MIN_ROWS = {"nasa_power_monthly": 120, "nasa_power": 3 * 365, "gpm": 365, "smap": 60, "modis": 24, "soilgrids": 1, "faostat": 20}
NR = "NON RÉALISÉE — aucune donnée réelle suffisante en base"


def n_rows(con, table, source):
    return con.execute(f"SELECT COUNT(*) FROM {table} WHERE source=?", (source,)).fetchone()[0]


def real_weather(con, region, source):
    d = pd.read_sql_query("SELECT * FROM weather WHERE region=? AND source=? ORDER BY date", con, params=(region, source))
    d["date"] = pd.to_datetime(d["date"]); return d


def regions_with(con, table, source):
    return [r[0] for r in con.execute(f"SELECT region FROM {table} WHERE source=? GROUP BY region HAVING COUNT(*)>=?", (source, MIN_ROWS[source]))]


def calib_power(con):
    out = []
    for reg in regions_with(con, "weather", "nasa_power"):
        r = real_weather(con, reg, "nasa_power").dropna(); R = REGIONS[reg]
        s = generate_weather(R, "1990-01-01", 365 * 30, 1)
        m = lambda d, c, f: d.groupby(d.date.dt.month)[c].agg(f)
        yrs = max(1, len(r) / 365)
        rain_b = 100 * (s.PRECTOTCORR.sum() / 30 / (r.PRECTOTCORR.sum() / yrs) - 1)
        out.append(f"{reg} : biais pluie annuelle simulée {rain_b:+.0f} % ; T {s.T2M.mean() - r.T2M.mean():+.1f} °C ; Rs {s.ALLSKY_SFC_SW_DWN.mean() - r.ALLSKY_SFC_SW_DWN.mean():+.1f} MJ/m²/j ; "
                   f"RH {s.RH2M.mean() - r.RH2M.mean():+.0f} points ; ET0 {et0_fao56(s, R.lat, R.elev).mean() * 365 / (et0_fao56(r.reset_index(drop=True), R.lat, R.elev).mean() * 365) * 100 - 100:+.0f} %  -> corriger les normales de regions.py")
    return out


def calib_soilgrids(con):
    d = pd.read_sql_query("SELECT * FROM soil WHERE source='soilgrids'", con)
    if d.empty: return []
    main = pd.read_sql_query("SELECT soc_pct, ph, cec, fc, wp FROM ds_main WHERE split='train'", con)
    out = []
    for c in ("soc_pct", "ph", "cec", "fc", "wp"):
        lo, hi = main[c].quantile([0.05, 0.95]); out.append(f"{c} : {100 * ((d[c] >= lo) & (d[c] <= hi)).mean():.0f} % des sites SoilGrids dans la plage 5-95 % d'entraînement")
    from core.mlkit import Predictor
    P = Predictor(os.path.join(ROOT, "models", "soil")); F = P.features
    X = d.assign(slope_pct=5.0, n_kg=0, p_kg=0, k_kg=0, lime_t=0, weeding=0.6, residue_return=0.3, rain_norm_season=700, rain_anom_prev60=1.0, rain_prev60=250)
    for c in [f for f in F if f.startswith("crop_")]: X[c] = 0.0
    X["crop_maize"] = 1.0
    out.append(f"{100 * P.predict(X[F]).ood_flag.mean():.0f} % des sols SoilGrids sont signalés « hors domaine » par Soil AI")
    return out


def calib_faostat(con):
    f = pd.read_sql_query("SELECT crop, year, yield_t_ha, flag FROM yield_reference WHERE source='faostat' AND year>=2010", con)
    if f.empty: return []
    m = pd.read_sql_query("SELECT crop, yield_t_ha FROM ds_main WHERE split='test' AND n_kg=0 AND p_kg=0 AND k_kg=0", con)
    share_e = float((f.flag == "E").mean()); out = []
    for c, g in f.groupby("crop"):
        s_ = m[m.crop == c].yield_t_ha
        if len(s_):
            lo, hi = s_.quantile([.1, .9]); fa = g.yield_t_ha.mean()
            out.append(f"{c} : FAOSTAT 2010+ {fa:.2f} t/ha (variation {g.yield_t_ha.min():.2f}-{g.yield_t_ha.max():.2f}) ; simulé sans engrais médiane {s_.median():.2f} (p10-p90 {lo:.2f}-{hi:.2f}) ; "
                       f"FAOSTAT {'dans' if lo <= fa <= hi else 'HORS'} l'intervalle simulé ; facteur FAOSTAT/simulé {fa / s_.median():.2f}")
    out.append(f"ATTENTION : {share_e:.0%} des valeurs FAOSTAT 2010+ sont des estimations FAO (drapeau E), quasi constantes d'une année à l'autre : "
               "elles donnent un NIVEAU national de référence, aucune information interannuelle ni locale ; un facteur de calage appliqué au jumeau serait une hypothèse")
    return out


def calib_power_monthly(con):
    """POWER mensuel (MERRA-2 / CERES : réanalyse et produit satellite-modèle, PAS des observations de terrain)."""
    out = []
    for reg in regions_with(con, "weather", "nasa_power_monthly"):
        m = re.match(r"pt_(-?[\d.]+)_(-?[\d.]+)", reg)
        lat, lon = (float(m.group(1)), float(m.group(2))) if m else (REGIONS[reg].lat, REGIONS[reg].lon) if reg in REGIONS else (None, None)
        if lat is None: continue
        near = min(REGIONS.values(), key=lambda r: (r.lat - lat) ** 2 + (r.lon - lon) ** 2)
        km = 111 * np.hypot(near.lat - lat, (near.lon - lon) * np.cos(np.radians(lat)))
        d = pd.read_sql_query("SELECT * FROM weather WHERE region=? AND source='nasa_power_monthly'", con, params=(reg,)); d["date"] = pd.to_datetime(d["date"])
        gen = generate_weather(near, "1990-01-01", 365 * 30, 1)
        out.append(f"{reg} : région simulée la plus proche = {near.name} (~{km:.0f} km) ; {len(d)} mois ({d.date.dt.year.min()}-{d.date.dt.year.max()}) ; comparaison indicative (sites différents)")
        for c, lab, u in (("T2M", "température", "°C"), ("ALLSKY_SFC_SW_DWN", "rayonnement", "MJ/m²/j"), ("RH2M", "humidité relative (dérivée de QV2M)", "pts"), ("WS2M", "vent à 2 m", "m/s")):
            if c in d and d[c].notna().any():
                rm = d.groupby(d.date.dt.month)[c].mean(); sm = gen.groupby(gen.date.dt.month)[c].mean()
                out.append(f"{lab} : réel {rm.mean():.2f} vs simulé {sm.mean():.2f} {u} (écart {sm.mean() - rm.mean():+.2f}) ; corrélation du cycle mensuel {np.corrcoef(rm, sm)[0, 1]:+.2f}")
        g = pd.read_sql_query("SELECT date, value FROM soil_moisture WHERE region=? AND kind='gwetroot_monthly' AND source='nasa_power_monthly'", con, params=(reg,))
        if len(g):
            g["date"] = pd.to_datetime(g["date"]); gm = g.groupby(g.date.dt.month).value.mean()
            soil = near.soils[0]; o = DigitalTwin(Site(near.name, soil)).run(gen.iloc[:365 * 10], [], seed=0, trace=True)["trace"]
            from core.regions import SOILS
            S_ = SOILS[soil]; sat = (S_.sat * 1000); rel = (o.W1 + o.W2) / sat
            sm_ = rel.groupby(pd.DatetimeIndex(gen.date.iloc[:365 * 10]).month.to_numpy()).mean()
            out.append(f"humidité de la zone racinaire : GWETROOT moyen {gm.mean():.2f} vs simulé (sol {soil}, nu) {sm_.mean():.2f} ; corrélation du cycle mensuel {np.corrcoef(gm, sm_)[0, 1]:+.2f}")
    return out


def ext_gpm(con):
    out = []
    for reg in regions_with(con, "weather", "gpm"):
        g = real_weather(con, reg, "gpm").set_index("date").PRECTOTCORR; p = real_weather(con, reg, "nasa_power").set_index("date").PRECTOTCORR
        j = pd.concat([g, p], axis=1, keys=["gpm", "power"]).dropna()
        if len(j) > 100:
            w = j.resample("W").sum(); out.append(f"{reg} : corrélation hebdomadaire GPM/POWER {w.corr().iloc[0, 1]:.2f} ; biais {100 * (w.power.sum() / max(w.gpm.sum(), 1e-9) - 1):+.0f} %")
    return out


def ext_smap(con):
    out = []
    for reg in regions_with(con, "soil_moisture", "smap"):
        w = real_weather(con, reg, "nasa_power").dropna()
        if len(w) < 365: continue
        sm = pd.read_sql_query("SELECT date, value FROM soil_moisture WHERE region=? AND source='smap'", con, params=(reg,)); sm["date"] = pd.to_datetime(sm["date"])
        soil = REGIONS[reg].soils[0]
        o = DigitalTwin(Site(reg, soil)).run(w.reset_index(drop=True), [], seed=0, trace=True)["trace"]
        o["date"] = w.date.reset_index(drop=True); o["theta"] = o.W1 / 300.0
        j = sm.merge(o[["date", "theta"]], on="date")
        if len(j) > 30: out.append(f"{reg} : corrélation de rang SMAP/simulé (sol nu, 0-30 cm) {spearmanr(j.value, j.theta)[0]:.2f} sur {len(j)} jours")
    return out


def ext_modis(con):
    out = []
    for reg in regions_with(con, "vegetation", "modis"):
        w = real_weather(con, reg, "nasa_power").dropna()
        if len(w) < 365: continue
        v = pd.read_sql_query("SELECT date, ndvi FROM vegetation WHERE region=? AND source='modis'", con, params=(reg,)); v["date"] = pd.to_datetime(v["date"])
        rain30 = w.set_index("date").PRECTOTCORR.rolling(30).sum().reset_index().rename(columns={"PRECTOTCORR": "rain30"})
        j = v.merge(rain30, on="date").dropna()
        if len(j) > 20: out.append(f"{reg} : corrélation de rang NDVI / pluie 30 j (forçage du jumeau) {spearmanr(j.ndvi, j.rain30)[0]:.2f} — indicateur indirect ; comparaison NDVI/couverture simulée par parcelle : à faire avec des parcelles géoréférencées")
    return out


def build(con):
    md = ["# Rapport de validation scientifique\n",
          "Trois catégories distinctes. **Aucune validation externe n'est déclarée sans données réelles en base.**\n"]
    # ---------- interne
    md.append("\n## 1. Validation interne (aucune donnée réelle)\n")
    tw = os.path.join(ROOT, "reports", "twin_checks.json")
    if os.path.exists(tw):
        t = pd.DataFrame(json.load(open(tw)))
        md.append(f"- Jumeau numérique : {(t.status == 'PASS').sum()}/{len(t)} contrôles de cohérence physique/agronomique passés (bilans eau/N/P/K fermés, réponses causales, stabilité, performance). Détail : `reports/twin_checks.md`.")
    for m in ("climate", "soil", "crop", "yield", "recommendation"):
        try:
            meta, _ = read_meta(os.path.join(ROOT, "models", m))
            md.append(f"- {m} : " + " ; ".join(f"{t} R²(test)={d['r2']['te']:.2f}, couverture 80 %={d['coverage_test']:.0%}" for t, d in meta["diagnostics"].items()))
        except Exception as e:
            md.append(f"- {m} : modèle non disponible ({type(e).__name__})")
    md.append("\n*Portée* : ces chiffres mesurent l'accord des modèles avec **le jumeau numérique** (vérité simulée), pas avec la réalité.\n")
    # ---------- calibration / externe
    src = [("NASA POWER (mensuel, MERRA-2/CERES)", "weather", "nasa_power_monthly", "Calibration", calib_power_monthly), ("NASA POWER (journalier)", "weather", "nasa_power", "Calibration", calib_power), ("SoilGrids", "soil", "soilgrids", "Calibration", calib_soilgrids),
           ("FAOSTAT", "yield_reference", "faostat", "Calibration", calib_faostat), ("GPM", "weather", "gpm", "Validation externe", ext_gpm),
           ("SMAP", "soil_moisture", "smap", "Validation externe", ext_smap), ("MODIS NDVI", "vegetation", "modis", "Validation externe", ext_modis)]
    for cat in ("Calibration", "Validation externe"):
        md.append(f"\n## {'2' if cat == 'Calibration' else '3'}. {cat}\n")
        md.append("*Ajuste des paramètres avec des données réelles : cela ne prouve rien d'indépendant.*\n" if cat == "Calibration" else
                  "*Observations indépendantes non utilisées pour ajuster le jumeau.*\n")
        for name, table, source, c, fn in src:
            if c != cat: continue
            n = n_rows(con, table, source)
            lines = fn(con) if n >= MIN_ROWS[source] else []
            md.append(f"- **{name}** ({table}, source `{source}`, {n} lignes) : " + ("\n  - " + "\n  - ".join(lines) if lines else NR))
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    con = connect(sys.argv[1] if len(sys.argv) > 1 else None)
    txt = build(con)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    open(os.path.join(ROOT, "reports", "validation_report.md"), "w").write(txt); print(txt)
