"""Audit empirique du jumeau numérique (cohérence physique/agronomique, bilans, sensibilités, performance). Chaque contrôle : valeur MESURÉE, plage attendue, verdict.
Usage : python scripts/audit_twin.py  (écrit audit_report.json + audit_report.md)"""
import sys, os, json, time, tracemalloc, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from core.regions import REGIONS, SOILS
from core.climate import generate_weather, et0_fao56, extraterrestrial_radiation
from core.twin import DigitalTwin, Site, Management
from core.events import Event
from core.schema import validate_weather, repair_weather, WEATHER_COLS

R = []   # (section, check, measured, expected, status)
def chk(sec, name, val, lo=None, hi=None, fmt="{:.3f}", note=""):
    ok = True
    if lo is not None and val < lo: ok = False
    if hi is not None and val > hi: ok = False
    exp = f"[{'' if lo is None else lo}, {'' if hi is None else hi}]"
    R.append(dict(section=sec, check=name, measured=fmt.format(val) if not isinstance(val, str) else val,
                  expected=exp, status="PASS" if ok else "FAIL", note=note))
def flag(sec, name, cond, note=""):
    R.append(dict(section=sec, check=name, measured=str(bool(cond)), expected="True", status="PASS" if cond else "FAIL", note=note))

LOW = dict(); HIGH = dict(n_kg=100, p_kg=30, k_kg=40, lime_t=1.0, pest_control=0.8, seed_quality=0.9, weeding=0.9)
def run(region, soil, plan, seeds=range(8), years=3, slope=5.0, events=(), crop_filter=None):
    ys = []
    for sd in seeds:
        w = generate_weather(REGIONS[region], "2000-01-01", 365 * years, sd)
        o = DigitalTwin(Site(region, soil, slope_pct=slope)).run(w, plan, events, seed=sd)
        d = o["seasons"]
        if len(d) and "skipped" in d: d = d[d.skipped == 0]
        if crop_filter is not None and len(d): d = d[d.crop == crop_filter]
        ys += list(d.yield_t_ha) if len(d) else []
    return np.array(ys)

# ---------------------------------------------------------------- 1. CLIMAT
S = "1 Climat"
for name, r in REGIONS.items():
    df = generate_weather(r, "1990-01-01", 365 * 30 + 8, 11)
    yrs = df.date.dt.year; ann = df.groupby(yrs).PRECTOTCORR.sum().iloc[:-1]
    mon = df.groupby(df.date.dt.month).PRECTOTCORR.sum() / 30
    chk(S, f"{name}: pluie annuelle vs cible (écart %)", 100 * (ann.mean() / sum(r.rain_mm) - 1), -8, 8, "{:+.1f}")
    chk(S, f"{name}: corrélation cycle mensuel", float(np.corrcoef(mon.values, r.rain_mm)[0, 1]), 0.95, None, note="seuil 0.95 : régimes équatoriaux à cycle plat = faible rapport signal/bruit")
    chk(S, f"{name}: CV interannuel pluie", ann.std() / ann.mean(), 0.07, 0.30)
    tm = df.groupby(df.date.dt.month).T2M.mean()
    chk(S, f"{name}: erreur T moyenne (°C, max mensuel)", float(np.abs(tm.values - np.array(r.tmean)).max()), None, 0.8)
    et = et0_fao56(df, r.lat, r.elev)
    chk(S, f"{name}: ET0 annuelle (mm)", et.sum() / 30.4, 1000, 1650, "{:.0f}", "plage large : à calibrer sur POWER")
    chk(S, f"{name}: RH moyenne (%)", df.RH2M.mean(), 55, 85, "{:.0f}")
    chk(S, f"{name}: fréquence jours humides", float((df.PRECTOTCORR > 0.1).mean()), 0.2, 0.55)
    qc = validate_weather(df)
    flag(S, f"{name}: QC physique (plages, Tmax>=Tmin, NaN)", qc["ok"])
    ra = extraterrestrial_radiation(r.lat, df.date.dt.dayofyear.to_numpy())
    flag(S, f"{name}: Rs <= 0.82 Ra", bool((df.ALLSKY_SFC_SW_DWN <= 0.82 * ra + 1e-9).all()))

# ---------------------------------------------------------------- 2. FERMETURE DES BILANS
S = "2 Bilans (conservation)"
worst = dict(w=0, wd=0, N=0, P=0, K=0)
evs = [(), (Event("flood", 300, 3, 120),), (Event("heatwave", 330, 8, 6),), (Event("fertility_loss", 0, 1, 0.5),), (Event("drought", 280, 40, 0.8),), (Event("pest", 300, 15, 0.4), Event("disease", 290, 40, 0.8))]
for k, ev in enumerate(evs):
    for soil, reg in (("arenosol_sand", "kinshasa"), ("andosol_loam", "bukavu"), ("fluvisol_loam", "mbandaka")):
        w = generate_weather(REGIONS[reg], "2000-01-01", 365 * 3, 20 + k)
        plan = [Management("maize", 0, 0, n_kg=60, p_kg=20, k_kg=20), Management("bean", 1, 0), Management("cassava", 2, 0) ] if reg != "mbandaka" else [Management("rice_upland", 0, 0), Management("groundnut", 1, 1)]
        b = DigitalTwin(Site(reg, soil, slope_pct=12)).run(w, plan, ev, seed=k)["balance"]
        worst["w"] = max(worst["w"], abs(b["water_resid_mm"])); worst["wd"] = max(worst["wd"], b["water_max_daily_resid"])
        for e in "NPK": worst[e] = max(worst[e], abs(b[f"{e}_resid"]))
chk(S, "résidu bilan hydrique cumulé (mm)", worst["w"], None, 1e-6, "{:.2e}")
chk(S, "résidu bilan hydrique journalier max (mm)", worst["wd"], None, 1e-6, "{:.2e}")
for e in "NPK": chk(S, f"résidu bilan {e} (kg/ha)", worst[e], None, 1e-6, "{:.2e}")

# ---------------------------------------------------------------- 3. RENDEMENTS vs plages de référence
S = "3 Rendements (plausibilité)"
# plages larges : essais/rendements attainables tropicaux (FAO, IITA, GYGA) — à remplacer par FAOSTAT/INERA
bench = {"maize": ((0.4, 2.8), (2.5, 7.0)), "cassava": ((4, 16), (10, 32)), "bean": ((0.3, 1.8), (1.0, 3.2)),
         "groundnut": ((0.3, 1.8), (1.0, 3.2)), "rice_upland": ((0.5, 2.8), (2.0, 5.5))}
for crop, (lo_b, hi_b) in bench.items():
    for lvl, kw, (a, b_) in (("faible intrant", dict(), lo_b), ("intrants élevés", HIGH, hi_b)):
        vals = []
        for reg, soil in (("kananga", "ferralsol_scl"), ("kinshasa", "ferralsol_scl"), ("kisangani", "acrisol_cl")):
            plan = [Management(crop, 0, 0, **kw)] + ([Management(crop, 1, 0, **kw)] if crop != "cassava" else [])
            vals.append(run(reg, soil, plan, range(6)).mean())
        chk(S, f"{crop} {lvl} (t/ha, moy. 3 sites ferralsol/acrisol)", float(np.mean(vals)), a, b_)
# écart station / exploitation
fl = run("kananga", "ferralsol_scl", [Management("maize", 0, 0, **HIGH)], range(10)).mean()
fp = run("kananga", "ferralsol_scl", [Management("maize", 0, 0)], range(10)).mean()
chk(S, "ratio exploitation faible intrant / intrants élevés (maïs)", fp / fl, 0.15, 0.6)

# ---------------------------------------------------------------- 4. SENSIBILITÉS PHYSIQUES
S = "4 Sensibilités (réponse causale)"
def paired(plan_a, plan_b, ev_a=(), ev_b=(), reg="kananga", soil="ferralsol_scl", seeds=range(12), slope=5.0, flt=None):
    a = run(reg, soil, plan_a, seeds, 3, slope, ev_a, flt); b = run(reg, soil, plan_b, seeds, 3, slope, ev_b, flt)
    k = min(len(a), len(b)); return a[:k], b[:k]
def n_resp():
    ys = [run("kananga", "ferralsol_scl", [Management("maize", 0, 0, n_kg=n, p_kg=20, k_kg=20)], range(10)).mean() for n in (0, 50, 100, 150)]
    return ys
ys = n_resp(); flag(S, "réponse N maïs monotone non décroissante (0/50/100/150)", all(np.diff(ys) >= -0.05), str([round(v, 2) for v in ys]))
ysP = [run("kananga", "ferralsol_scl", [Management("maize", 0, 0, n_kg=80, p_kg=p, k_kg=20)], range(10)).mean() for p in (0, 15, 30)]
flag(S, "réponse P maïs monotone (0/15/30) sol P-déficient", all(np.diff(ysP) >= -0.05), str([round(v, 2) for v in ysP]))
base = [Management("maize", 0, 0, n_kg=60, p_kg=20, k_kg=20)]
# événements ancrés sur le semis : on cible l'octobre de l'an 0 (jour ~ 280-340) pour Kananga
a, b = paired(base, base, (), (Event("drought", 285, 60, 0.85),))
chk(S, "sécheresse (-85 % pluie 60 j) : variation rendement maïs %", 100 * (b.mean() / a.mean() - 1), None, -10, "{:+.1f}")
a, b = paired(base, base, (), (Event("flood", 300, 4, 120),))
chk(S, "inondation (4 j x 120 mm) : variation rendement maïs %", 100 * (b.mean() / a.mean() - 1), None, -5, "{:+.1f}")
a, b = paired(base, base, (), (Event("heatwave", 325, 15, 8),))
chk(S, "canicule (+8 °C, 15 j en floraison) : variation %", 100 * (b.mean() / a.mean() - 1), None, -3, "{:+.1f}")
a, b = paired(base, base, (), (Event("fertility_loss", 0, 1, 0.5),))
chk(S, "perte de fertilité -50 % : variation %", 100 * (b.mean() / a.mean() - 1), None, -10, "{:+.1f}")
a, b = paired(base, base, (), (Event("pest", 300, 15, 0.5),), flt="maize")
chk(S, "ravageur (50 % sur 15 j) : variation %", 100 * (b.mean() / a.mean() - 1), None, -3, "{:+.1f}")
a, b = paired(base, base, (), (Event("disease", 290, 80, 1.0),), flt="maize")
chk(S, "épidémie (x5 infection) : variation %", 100 * (b.mean() / a.mean() - 1), None, -1, "{:+.1f}")
def eros(slope, res=0.3, crop="maize"):
    es = []
    for sd in range(8):
        w = generate_weather(REGIONS["bukavu"], "2000-01-01", 365 * 2, sd)
        o = DigitalTwin(Site("bukavu", "nitisol_clay", slope_pct=slope)).run(w, [Management(crop, 0, 0, residue_return=res)], seed=sd)
        es.append(o["balance"]["eros_total_t_ha"] / 2)
    return np.mean(es)
e5, e25 = eros(5), eros(25)
chk(S, "érosion pente 25 % / pente 5 % (ratio)", e25 / e5, 2.0, None)
chk(S, "érosion annuelle pente 25 % (t/ha/an)", e25, 5, 250, "{:.1f}", "ordre de grandeur pentes cultivées Grands Lacs ; NON validé")
# rotation : maïs après haricot vs maïs après maïs (même sol, N=0)
def rot(prev):
    out = []
    for sd in range(24):
        w = generate_weather(REGIONS["kinshasa"], "2000-01-01", 365 * 2, sd)
        plan = [Management(prev, 0, 1, residue_return=0.8), Management("maize", 0, 0, residue_return=0.8)]   # saison B puis saison A
        d = DigitalTwin(Site("kinshasa", "ferralsol_scl")).run(w, plan, seed=sd)["seasons"]
        d = d[d.skipped == 0] if "skipped" in d else d
        m = d[d.crop == "maize"]
        if len(m): out.append(m.yield_t_ha.iloc[-1])
    return np.array(out)
rm, rb = rot("maize"), rot("bean")
chk(S, "effet précédent légumineuse sur maïs (%)", 100 * (rb.mean() / rm.mean() - 1), 3, 60, "{:+.1f}", "littérature : +10 à +40 %")

# ---------------------------------------------------------------- 5. SOL À LONG TERME
S = "5 Dynamique du sol (10 ans)"
w = generate_weather(REGIONS["kananga"], "2000-01-01", 365 * 10, 5)
plan = [Management("maize", y, 0, residue_return=0.1) for y in range(10)]
o = DigitalTwin(Site("kananga", "ferralsol_scl", slope_pct=8)).run(w, plan, seed=5)
d = o["seasons"]; d = d[d.skipped == 0] if "skipped" in d else d
soc0 = SOILS["ferralsol_scl"].soc; soc1 = d.soc_end_pct.iloc[-1]
chk(S, "perte relative de COS en 10 ans, bas intrants (%/an)", 100 * (soc1 / soc0 - 1) / 10, -6, 0.2, "{:+.2f}")
chk(S, "variation pH en 10 ans (unités)", float(d.ph_end.iloc[-1] - SOILS['ferralsol_scl'].ph), -1.0, 0.1, "{:+.2f}")
chk(S, "cumul érosion 10 ans (t/ha)", o["balance"]["eros_total_t_ha"], 5, 600, "{:.0f}")
plan = [Management("maize", y, 0, n_kg=80, p_kg=20, k_kg=20, residue_return=1.0) for y in range(10)]
o2 = DigitalTwin(Site("kananga", "ferralsol_scl", slope_pct=8)).run(w, plan, seed=5)
d2 = o2["seasons"]; d2 = d2[d2.skipped == 0] if "skipped" in d2 else d2
chk(S, "COS fin : (résidus 100 % + engrais) - (résidus 10 %), % points", float(d2.soc_end_pct.iloc[-1] - soc1), 0.0, None, "{:+.3f}")

# ---------------------------------------------------------------- 6. DÉTERMINISME / ROBUSTESSE ENTRÉES
S = "6 Robustesse des entrées"
w = generate_weather(REGIONS["kinshasa"], "2000-01-01", 365 * 2, 4)
pl = [Management("maize", 0, 0, n_kg=50)]
y1 = DigitalTwin(Site("kinshasa", "ferralsol_scl")).run(w, pl, seed=9)["seasons"].yield_t_ha.iloc[0]
y2 = DigitalTwin(Site("kinshasa", "ferralsol_scl")).run(w, pl, seed=9)["seasons"].yield_t_ha.iloc[0]
flag(S, "déterminisme (même graine => même résultat)", y1 == y2)
wn = w.copy(); wn.loc[100:104, "PRECTOTCORR"] = np.nan
try:
    DigitalTwin(Site("kinshasa", "ferralsol_scl")).run(wn, pl); flag(S, "refus des NaN non réparés", False)
except ValueError:
    flag(S, "refus des NaN non réparés (pas de fabrication silencieuse)", True)
wr, rep = repair_weather(wn, 5)
flag(S, "réparation des trous <= 5 j puis simulation OK", wr[WEATHER_COLS].isna().sum().sum() == 0 and
     len(DigitalTwin(Site("kinshasa", "ferralsol_scl")).run(wr, pl)["seasons"]) > 0)
wo = w.copy(); wo.loc[50, "PRECTOTCORR"] = 900.0
flag(S, "valeur aberrante (900 mm/j) détectée par le QC", not validate_weather(wo)["ok"])
# bruit capteur : bruit 10 % pluie, ±1°C -> variabilité rendement bornée
rng = np.random.default_rng(0); ys = []
for k in range(15):
    wn2 = w.copy(); wn2["PRECTOTCORR"] *= rng.lognormal(0, 0.10, len(w)); wn2["T2M_MAX"] += rng.normal(0, 1, len(w)); wn2["T2M_MIN"] += rng.normal(0, 1, len(w))
    wn2["T2M_MAX"] = np.maximum(wn2["T2M_MAX"], wn2["T2M_MIN"] + 0.5); wn2["T2M"] = 0.5 * (wn2["T2M_MAX"] + wn2["T2M_MIN"])
    ys.append(DigitalTwin(Site("kinshasa", "ferralsol_scl")).run(wn2, pl, seed=9)["seasons"].yield_t_ha.iloc[0])
chk(S, "CV du rendement sous bruit capteur (pluie 10 %, T ±1 °C)", float(np.std(ys) / np.mean(ys)), None, 0.25)

# ---------------------------------------------------------------- 8. PERFORMANCE
S = "8 Performance"
w = generate_weather(REGIONS["kananga"], "2000-01-01", 365 * 3, 1)
tw = DigitalTwin(Site("kananga", "ferralsol_scl")); pl = [Management("maize", 0, 0), Management("bean", 1, 0), Management("maize", 2, 0)]
t = time.perf_counter(); N = 30
for k in range(N): tw.run(w, pl, seed=k)
dt = (time.perf_counter() - t) / N
chk(S, "temps par simulation de 3 ans (ms)", dt * 1000, None, 300, "{:.0f}")
chk(S, "saisons simulées par seconde (1 cœur)", 3 / dt, 10, None, "{:.0f}")
tracemalloc.start(); tw.run(w, pl, seed=1); peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
chk(S, "pic mémoire par simulation (Mo)", peak, None, 50, "{:.1f}")
chk(S, "durée pour 1 million de saisons, 1 cœur (heures)", 1e6 / (3 / dt) / 3600, None, 40, "{:.1f}", "parallélisable (batch.py)")

# ---------------------------------------------------------------- 9. STABILITÉ NUMÉRIQUE (cas extrêmes)
S = "9 Stabilité numérique"
bad = nh = n9 = 0; wst = 0.0
ex = [(Event("flood", 300, 5, 400),), (Event("heatwave", 320, 20, 15),), (Event("fertility_loss", 0, 1, 0.98),), (Event("drought", 280, 90, 0.99),),
      (Event("pest", 300, 20, 0.95), Event("disease", 290, 80, 1.0), Event("flood", 330, 3, 300))]
for reg, soil in (("kinshasa", "arenosol_sand"), ("bukavu", "andosol_loam"), ("lubumbashi", "ferralsol_scl"), ("mbandaka", "fluvisol_loam")):
    for ev in ex:
        for crop in ("maize", "cassava", "bean", "groundnut", "rice_upland"):
            w = generate_weather(REGIONS[reg], "2000-01-01", 1095, 3)
            o = DigitalTwin(Site(reg, soil, slope_pct=35)).run(w, [Management(crop, 1, 0, n_kg=150, p_kg=40, k_kg=60)], ev, seed=1, trace=True)
            t = o["trace"][["W1", "W2", "N_min", "P_lab", "K_ex", "CC", "B", "Ks", "Kw", "f_nut"]]; y = o["seasons"]["yield_t_ha"].dropna() if "yield_t_ha" in o["seasons"] else pd.Series(dtype=float)
            nh += int(len(y) == 0); n9 += 1
            bad += int((t < -1e-9).any().any() or not np.isfinite(t.to_numpy()).all() or ((y < 0) | (y > 60)).any())
            wst = max(wst, *(abs(o["balance"][k]) for k in ("water_resid_mm", "N_resid", "P_resid", "K_resid")))
chk(S, f"états négatifs / non finis / rendement aberrant sur {n9} simulations extrêmes (pente 35 %)", bad, None, 0, "{:.0f}")
chk(S, "résidu de bilan maximal en conditions extrêmes (mm ou kg/ha)", wst, None, 1e-6, "{:.1e}")
R.append(dict(section=S, check="simulations extrêmes sans récolte (culture non arrivée à maturité dans l'horizon)", measured=f"{nh}/{n9}", expected="", status="INFO", note="à interpréter : cycle trop long (manioc en altitude) ou semis impossible"))

# ---------------------------------------------------------------- sortie
df = pd.DataFrame(R)
df.to_json("reports/twin_checks.json", orient="records", force_ascii=False, indent=1)
with open("reports/twin_checks.md", "w") as f:
    for sec, g in df.groupby("section", sort=True):
        f.write(f"\n### {sec}  ({(g.status=='PASS').sum()}/{len(g)} PASS)\n\n| contrôle | mesuré | attendu | statut |\n|---|---|---|---|\n")
        for _, r in g.iterrows(): f.write(f"| {r.check} | {r.measured} | {r.expected} | {r.status} |\n")
print(df.groupby("section").status.value_counts().unstack(fill_value=0))
print("\nÉCHECS :")
for _, r in df[df.status == "FAIL"].iterrows(): print(f" - [{r.section}] {r.check}: mesuré {r.measured} attendu {r.expected} {r.note}")
