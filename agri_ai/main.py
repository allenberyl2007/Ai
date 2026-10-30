"""Point d'entrée UNIQUE. Appels successifs : Climate AI -> Soil AI -> Crop AI -> Yield AI -> Recommendation AI.
Fonctionne hors ligne (SQLite + modèles ONNX locaux).
Exemple : python main.py --region kananga --soil ferralsol_scl --slope 8 --prev bean --demo"""
import sys, os, math, json, argparse, importlib.util, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from core.db import connect, read_weather, put
from core.regions import REGIONS, SEASONS
from core.climate import generate_weather
from core.features import pre_season_features
from core.policy import CROPS, FR
from core.explain import explain
from core.decide import decide


def load_model(name):
    spec = importlib.util.spec_from_file_location(f"m_{name}", os.path.join(HERE, "models", name, "predict.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def get_soil(con, site, sources):
    for s in sources:
        r = pd.read_sql_query("SELECT * FROM soil WHERE site=? AND source=?", con, params=(site, s))
        if len(r): return r.iloc[0].to_dict(), s
    raise KeyError(f"sol '{site}' absent de la base (sources essayées : {sources}) — lancez scripts/create_db.py")


def planning_index(w, region, season, date=None):
    sd, _ = SEASONS[region][min(season, len(SEASONS[region]) - 1)]
    doy = pd.DatetimeIndex(w["date"]).dayofyear.to_numpy()
    if date:
        idx = int(np.where(pd.DatetimeIndex(w["date"]) == pd.Timestamp(date))[0][0])
    else:
        c = np.where(doy == sd)[0]; c = c[c >= 60]
        if len(c) == 0: raise ValueError("pas assez de météo (60 jours avant la fenêtre de semis) dans la base")
        idx = int(c[-1])
    return idx


def recommend(region, soil_site, slope, mgmt, prev, season=0, mode="sim", date=None, weather=None, verbose=True):
    con = connect()
    srcs_w = ("nasa_power", "sim") if mode == "real" else ("sim",)
    srcs_s = ("soilgrids", "archetype") if mode == "real" else ("archetype",)
    notes, data_factor = [], 1.0
    w = weather if weather is not None else read_weather(con, region, "1990-01-01", "2100-01-01", srcs_w)
    if w is None or len(w) < 120:
        raise ValueError("Pas de météo en base pour cette région. Utilisez --demo (météo simulée) ou scripts/download_all.py.")
    idx = planning_index(w, region, season, date)
    pre = pre_season_features(w.reset_index(drop=True), region, idx)
    real_used = bool(mode == "real" and "nasa_power" in set(w.get("source", [])))
    soil, soil_src = get_soil(con, soil_site, srcs_s)
    if soil_src == "soilgrids": notes.append("Phosphore et potassium du sol estimés (non mesurés par satellite).")
    soil = {k: soil[k] for k in ("soc_pct", "ph", "p_bray", "k_exch", "cec", "wp", "fc", "sat", "drain", "cn", "k_erod")}
    soil["slope_pct"] = slope
    rows = []
    for c in CROPS:
        r = dict(soil, **mgmt, **pre, prev_legume=float(prev in ("bean", "groundnut")), prev_none=float(prev is None), prev_same=float(prev == c))
        r.update({f"crop_{k}": float(k == c) for k in CROPS}); rows.append(r)
    X = pd.DataFrame(rows, index=CROPS)

    M = {n: load_model(n) for n in ("climate", "soil", "crop", "yield", "recommendation")}
    clim = M["climate"].predict(X.iloc[[0]])                         # 1) Climate AI
    so = M["soil"].predict(X)                                        # 2) Soil AI
    cr = M["crop"].predict(X)                                        # 3) Crop AI
    yl = M["yield"].predict(X)                                       # 4) Yield AI
    R = pd.DataFrame(index=CROPS)                                    # 5) Recommendation AI
    for t in ("season_rain_mm", "dry_spell_days", "max3_rain_mm", "heat_days"): R[t] = float(clim[f"{t}_mid"].iloc[0])
    for t in ("erosion_t_ha", "d_soc_pct", "d_ph", "n_leach_kg"): R[t] = so[f"{t}_mid"]
    for t in ("days_to_harvest", "water_use_mm", "water_stress", "n_uptake_kg"): R[t] = cr[f"{t}_mid"]
    R["yield_t_ha"] = yl["yield_t_ha_mid"]
    for c in CROPS: R[f"crop_{c}"] = X[f"crop_{c}"]
    rc = M["recommendation"].predict(R)

    clim_conf = float(clim[[f"{t}_conf" for t in ("season_rain_mm", "dry_spell_days", "max3_rain_mm", "heat_days")]].mean(axis=1).iloc[0])
    real_used = bool(real_used)
    dc = decide(rc, so, cr, yl, clim_conf, real_used)
    order, top, alt, p_top, overall, conf = dc["order"], dc["top"], dc["alt"], dc["p_top"], dc["overall"], dc["conf"]
    if not real_used: notes.append("Le système n'a été entraîné que sur des simulations : confiance limitée.")
    ood = bool(so.loc[top, "ood_flag"] or cr.loc[top, "ood_flag"] or yl.loc[top, "ood_flag"] or rc.loc[top, "ood_flag"] or clim["ood_flag"].iloc[0])
    unseen = sorted(set(sum([list(so.loc[top, "unseen_features"]), list(yl.loc[top, "unseen_features"]), list(cr.loc[top, "unseen_features"])], [])))
    if ood: notes.append("Situation inhabituelle, différente de ce que le système a appris" + (f" ({', '.join(unseen)})" if unseen else "") + " : confiance réduite.")
    if p_top < 0.65: notes.append(f"{FR[top]} et {FR[alt]} sont très proches : les deux sont de bons choix.")
    cl = {t: (float(clim[f"{t}_lo"].iloc[0]), float(clim[f"{t}_mid"].iloc[0]), float(clim[f"{t}_hi"].iloc[0])) for t in ("dry_spell_days", "season_rain_mm", "max3_rain_mm")}
    g = lambda df, c, t: (df.loc[c, f"{t}_lo"], df.loc[c, f"{t}_mid"], df.loc[c, f"{t}_hi"])
    reasons = explain(top, soil, mgmt, cl, {"erosion_t_ha": g(so, top, "erosion_t_ha")}, {"water_use_mm": g(cr, top, "water_use_mm")}, None, pre["rain_norm_season"])
    y = g(yl, top, "yield_t_ha")
    if y[1] < (3.0 if top == "cassava" else 0.4):
        reasons.append("Attention : la récolte attendue est très faible, même pour la meilleure culture. Améliorer le sol (engrais, chaux, compost) aiderait beaucoup.")
    res = dict(region=region, crop=top, crop_fr=FR[top], alternative=alt, alternative_fr=FR[alt], confidence=round(conf, 2),
               yield_t_ha=dict(low=round(y[0], 2), mid=round(y[1], 2), high=round(y[2], 2)), reasons=reasons, notes=notes,
               ranking=[dict(crop=c, score=round(float(rc.loc[c, "score_ref_mid"]), 2), conf=round(float(overall[c]), 2)) for c in order],
               planning_date=str(pd.Timestamp(w["date"].iloc[idx]).date()))
    put(con, "recommendation", pd.DataFrame([dict(ts=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), region=region, crop=top, score=float(rc.loc[top, "score_ref_mid"]),
                                                  confidence=conf, payload=json.dumps(res, ensure_ascii=False, default=float))]), "real" if real_used else "sim")
    if verbose: print(render(res))
    return res


def render(r):
    pct = int(round(r["confidence"] * 100))
    lines = [f"Culture recommandée :\n\n{r['crop_fr']}\n", "Pourquoi ?\n"] + [f"{x}" for x in r["reasons"]]
    y = r["yield_t_ha"]
    lines += [f"Récolte attendue : environ {y['mid']:.1f} tonnes par hectare (entre {y['low']:.1f} et {y['high']:.1f}).", "",
              f"Confiance :\n\n{pct} %", "", f"Alternative :\n\n{r['alternative_fr']}"]
    if pct < 50: lines += ["", "Réponse peu sûre : demandez aussi conseil à un technicien agricole."]
    lines += [""] + [f"Note : {n}" for n in r["notes"]]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=list(REGIONS)); ap.add_argument("--soil", help="nom du sol dans la table soil (défaut : archétype)")
    ap.add_argument("--slope", type=float, default=5.0); ap.add_argument("--season", type=int, default=0)
    ap.add_argument("--prev", default="none", choices=["none", "maize", "bean", "groundnut", "rice_upland"])
    ap.add_argument("--n", type=float, default=0); ap.add_argument("--p", type=float, default=0); ap.add_argument("--k", type=float, default=0)
    ap.add_argument("--lime", type=float, default=0); ap.add_argument("--weeding", type=float, default=0.6)
    ap.add_argument("--residue", type=float, default=0.3); ap.add_argument("--pest-control", type=float, default=0.0)
    ap.add_argument("--seed-quality", type=float, default=0.5); ap.add_argument("--mode", default="sim", choices=["sim", "real"])
    ap.add_argument("--date"); ap.add_argument("--demo", action="store_true", help="génère 3 ans de météo simulée pour la région")
    ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    mg = dict(n_kg=a.n, p_kg=a.p, k_kg=a.k, lime_t=a.lime, weeding=a.weeding, residue_return=a.residue, pest_control=a.pest_control, seed_quality=a.seed_quality)
    wdf = None
    if a.demo:
        wdf = generate_weather(REGIONS[a.region], "2000-01-01", 1095, 4242); wdf["source"] = "sim"
        c = connect(); d = wdf.copy(); d.insert(0, "region", a.region); put(c, "weather", d.drop(columns="source"), "sim")
        wdf = wdf.iloc[:365 * 2 + 120]       # « aujourd'hui » = début de la 3e année
    r = recommend(a.region, a.soil or REGIONS[a.region].soils[0], a.slope, mg, None if a.prev == "none" else a.prev, a.season, a.mode, a.date, wdf, verbose=not a.json)
    if a.json: print(json.dumps(r, ensure_ascii=False, indent=1, default=float))
