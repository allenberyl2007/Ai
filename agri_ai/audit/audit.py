"""Audit automatique : architecture, données, machine learning, jumeau numérique, validation scientifique.
Usage : python audit/audit.py   ->  reports/audit_report.md + reports/audit_report.json"""
import sys, os, ast, json, glob, subprocess, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from core.db import connect
from core.mlkit import Predictor, prob_exceed, _fit_heads, CONFIGS, _r2, read_meta
from core.decide import decide
from core.policy import CROPS
from core.features import pre_season_features
from core.regions import REGIONS
from core.climate import generate_weather

MODELS = ["climate", "soil", "crop", "yield", "recommendation"]
R = []
def add(sec, name, measured, expected="", status="INFO", note=""):
    R.append(dict(section=sec, check=name, measured=str(measured), expected=expected, status=status, note=note))
def chk(sec, name, val, lo=None, hi=None, fmt="{:.3f}", note="", warn=False):
    ok = (lo is None or val >= lo) and (hi is None or val <= hi)
    add(sec, name, fmt.format(val), f"[{'' if lo is None else lo}, {'' if hi is None else hi}]", "PASS" if ok else ("WARN" if warn else "FAIL"), note)
def flag(sec, name, cond, note="", warn=False):
    add(sec, name, bool(cond), "True", "PASS" if cond else ("WARN" if warn else "FAIL"), note)

con = connect()
main_df = pd.read_sql_query("SELECT * FROM ds_main", con); clim_df = pd.read_sql_query("SELECT * FROM ds_climate", con)
meta = {m: read_meta(f"models/{m}")[0] for m in MODELS}

# =============================================================== 1. ARCHITECTURE
S = "1 Architecture"
need = ["main.py"] + [f"models/{m}/{f}" for m in MODELS for f in ("train.py", "predict.py", "model.onnx")]
flag(S, "fichiers requis (main.py + 5 x train.py/predict.py/model.onnx)", all(os.path.exists(f) for f in need), str([f for f in need if not os.path.exists(f)]))
flag(S, "chaque dossier modèle contient UNIQUEMENT train.py, predict.py, model.onnx", all(sorted(f for f in os.listdir(f"models/{m}") if f != "__pycache__") == ["model.onnx", "predict.py", "train.py"] for m in MODELS), str({m: sorted(os.listdir(f"models/{m}")) for m in MODELS}))
flag(S, "aucun Docker / Kubernetes / compose", not any(os.path.exists(f) for f in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml")) and not glob.glob("**/k8s*", recursive=True))
pyfiles = [f for f in glob.glob("**/*.py", recursive=True) if not f.startswith(("tests/", "audit/"))]
bad = {"flask", "fastapi", "django", "uvicorn", "celery", "kubernetes", "docker", "aiohttp", "grpc", "pika", "redis", "starlette"}
imps = set()
for f in pyfiles:
    for n in ast.walk(ast.parse(open(f).read())):
        if isinstance(n, ast.Import): imps |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0: imps.add(n.module.split(".")[0])
flag(S, "aucun serveur / microservice / API interne (imports)", not (imps & bad), str(imps & bad))
flag(S, "une seule base SQLite", len(glob.glob("data/*.db")) == 1, str(glob.glob("data/*.db")))
tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
flag(S, "7 tables requises + datasets", {"weather", "soil", "soil_moisture", "vegetation", "yield_reference", "crop", "recommendation", "ds_main", "ds_climate"} <= tabs)
flag(S, "chaque table de données garde sa source", all("source" in [c[1] for c in con.execute(f"pragma table_info({t})")] for t in ("weather", "soil", "soil_moisture", "vegetation", "yield_reference", "crop", "recommendation")))
funcs = []
for f in pyfiles:
    t = ast.parse(open(f).read())
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef,)):
            cc = 1 + sum(isinstance(x, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp, ast.IfExp, ast.comprehension)) for x in ast.walk(n))
            funcs.append((f, n.name, n.end_lineno - n.lineno + 1, cc))
top = sorted(funcs, key=lambda x: -x[3])[:3]
add(S, "complexité cyclomatique approx. : 3 plus grosses fonctions", "; ".join(f"{a}:{b}={d}" for a, b, c, d in top), "", "INFO",
    "twin.run() est un bloc monolithique (boucle journalière) : lisible mais long ; à découper si maintenance difficile")
chk(S, "complexité max d'une fonction hors twin.run", max(d for a, b, c, d in funcs if not (a.endswith("twin.py") and b == "run")), None, 60, "{:.0f}", warn=True)
chk(S, "longueur du plus gros fichier (lignes)", max(len(open(f).read().splitlines()) for f in pyfiles), None, 500, "{:.0f}", warn=True)
flag(S, "chaque predict.py <= 25 lignes", all(len(open(f"models/{m}/predict.py").read().splitlines()) <= 25 for m in MODELS))
indep = all(not any(("models" in l and "import" in l) for l in open(f"models/{m}/predict.py")) for m in MODELS)
flag(S, "modèles indépendants (aucun predict.py n'importe un autre modèle)", indep)
flag(S, "chaque train.py lit uniquement ses colonnes (SELECT explicite, jamais SELECT *)", all("SELECT *" not in open(f"models/{m}/train.py").read() and "SELECT" in open(f"models/{m}/train.py").read() for m in MODELS))
flag(S, "aucun train.py ne lit un autre modèle (entraînement indépendant)", all(not any(k in open(f"models/{m}/train.py").read() for k in ("meta.json", "model.onnx", "read_meta", "Predictor")) for m in MODELS))
flag(S, "docstring de module présente (lisibilité)", all(ast.get_docstring(ast.parse(open(f).read())) for f in pyfiles if not f.endswith("__init__.py") and "predict.py" not in f), warn=True)
code = ("import socket,sys,runpy\ndef _b(*a,**k): raise OSError('réseau bloqué')\n"
        "socket.socket.connect=_b; socket.create_connection=_b; socket.getaddrinfo=_b\n"
        "sys.argv=['main.py','--region','kananga','--slope','8','--demo','--json']\nrunpy.run_path('main.py',run_name='__main__')")
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
try:
    out = json.loads(p.stdout); flag(S, "fonctionnement HORS LIGNE (réseau bloqué, main.py --demo)", p.returncode == 0 and "crop" in out, f"culture={out.get('crop')}, confiance={out.get('confidence')}")
except Exception:
    flag(S, "fonctionnement HORS LIGNE (réseau bloqué, main.py --demo)", False, (p.stderr or p.stdout)[-300:])

# =============================================================== 2. DONNÉES
S = "2 Données"
sp = lambda df, c: {s: set(df[df.split == s][c]) for s in ("train", "val", "test")}
for c in ("site_id", "wseed", "pair_id"):
    d = sp(main_df, c); inter = len(d["train"] & d["val"]) + len(d["train"] & d["test"]) + len(d["val"] & d["test"])
    flag(S, f"aucun {c} partagé entre train/val/test (ds_main)", inter == 0, f"intersections={inter}")
dc, dm = sp(clim_df, "wseed"), sp(main_df, "wseed")
cross = sum(len(dc[a] & dm[b]) for a in dc for b in dm if a != b)
flag(S, "graines météo disjointes entre splits ET entre ds_climate/ds_main", sum(len(dc[a] & dc[b]) for a in dc for b in dc if a < b) == 0 and cross == 0)
all_targets = set(sum([meta[m]["targets"] for m in MODELS], []))
for m in ("climate", "soil", "crop", "yield"):
    F = set(meta[m]["features"])
    flag(S, f"{m}: aucune cible d'un autre modèle parmi ses variables d'entrée", not (F & all_targets))
    flag(S, f"{m}: aucune variable « future » (préfixe clim_, cibles)", not any(f.startswith("clim_") for f in F))
    d = main_df if m != "climate" else clim_df
    X = d[meta[m]["features"]]
    chk(S, f"{m}: variables constantes", int((X.std() < 1e-9).sum()), None, 0, "{:.0f}")
    base = [c for c in X.columns if not c.startswith(("crop_", "prev_"))]
    cm = X[base].corr().abs().where(np.triu(np.ones((len(base), len(base))), 1).astype(bool))
    hi = [(a, b) for a in cm.index for b in cm.columns if cm.loc[a, b] > 0.98]
    add(S, f"{m}: paires de variables quasi identiques (|corr|>0.98)", len(hi), "0", "PASS" if not hi else "WARN", str(hi[:4]))
flag(S, "recommendation: entrées = sorties des 4 autres modèles uniquement (+ culture)", set(meta["recommendation"]["features"]) <= all_targets | {f"crop_{c}" for c in CROPS})
# fuite temporelle : les variables « avant saison » ne changent pas si on modifie le futur
r = REGIONS["kananga"]; w = generate_weather(r, "2000-01-01", 1095, 5); idx = 365 + 268 - 1
f0 = pre_season_features(w, "kananga", idx); w2 = w.copy(); w2.iloc[idx:, 1:] = generate_weather(r, "2000-01-01", 1095, 999).iloc[idx:, 1:].to_numpy()
f1 = pre_season_features(w2, "kananga", idx)
flag(S, "cohérence temporelle : variables pré-saison insensibles à la météo future (test de remplacement)", all(abs(f0[k] - f1[k]) < 1e-9 for k in f0))
dup = int(main_df.duplicated(subset=[c for c in main_df.columns if c not in ("pair_id", "site_id", "wseed")]).sum())
chk(S, "lignes dupliquées (ds_main)", dup, None, 0, "{:.0f}")
Fy = meta["yield"]["features"]; tr_keys = set(map(tuple, main_df[main_df.split == "train"][Fy].round(4).to_numpy()))
te_keys = list(map(tuple, main_df[main_df.split == "test"][Fy].round(4).to_numpy()))
chk(S, "vecteurs d'entrée de test déjà vus en train (contamination)", sum(k in tr_keys for k in te_keys), None, 0, "{:.0f}")
bounds = dict(soc_pct=(0.2, 10), ph=(3.5, 8), p_bray=(0.5, 60), k_exch=(0.02, 2), cec=(1, 50), wp=(0.02, 0.35), fc=(0.05, 0.5), sat=(0.2, 0.65),
              slope_pct=(0, 60), yield_t_ha=(0, 40), days_to_harvest=(30, 430), water_stress=(0, 1), erosion_t_ha=(0, 1000), rain_prev60=(0, 900),
              season_rain_mm=(0, 1500), lat=(-14, 6), elev=(0, 3000))
viol = {k: int(((main_df[k] < lo) | (main_df[k] > hi)).sum()) for k, (lo, hi) in bounds.items() if k in main_df}
chk(S, "unités/plages : valeurs hors plages physiques", sum(viol.values()), None, 0, "{:.0f}", str({k: v for k, v in viol.items() if v}))
add(S, "valeurs manquantes (ds_main, ds_climate)", f"{int(main_df.isna().sum().sum())}, {int(clim_df.isna().sum().sum())}", "0, 0",
    "PASS" if main_df.isna().sum().sum() + clim_df.isna().sum().sum() == 0 else "FAIL")
chk(S, "saisons avec érosion extrême (> 300 t/ha)", int((main_df.erosion_t_ha > 300).sum()), None, 0, "{:.0f}", "tempête exceptionnelle sur forte pente : physiquement possible mais rare ; à surveiller", warn=True)
for t in ("yield_t_ha", "erosion_t_ha", "water_use_mm", "season_rain_mm"):
    d = main_df[t] if t in main_df else clim_df[t]; mad = (d - d.median()).abs().median() + 1e-9
    chk(S, f"valeurs aberrantes {t} (> 8 MAD, %)", 100 * float(((d - d.median()).abs() > 8 * mad).mean()), None, 5, "{:.1f}", "les queues lourdes sont physiques (érosion, pluies) : conservées, interprétées par les intervalles", warn=True)
cnt = main_df.groupby("crop").size()
chk(S, "équilibre des lignes par culture (min/max)", float(cnt.min() / cnt.max()), 0.8, None)
best = main_df.loc[main_df.groupby("pair_id").score_ref.idxmax()].crop.value_counts(normalize=True)
add(S, "équilibre des classes de décision (meilleure culture de référence)", ", ".join(f"{k} {v:.0%}" for k, v in best.items()), "chaque classe >= 3 %, aucune > 60 %",
    "PASS" if (best.min() >= 0.03 and best.max() <= 0.60) else "WARN",
    "déséquilibre issu de la règle de valeur (policy.py) : le riz pluvial est presque jamais optimal ; ne pas évaluer l'IA par l'exactitude brute mais par le regret")
for s_ in ("train", "val", "test"):
    add(S, f"régions présentes dans {s_}", main_df[main_df.split == s_].region.nunique(), "6", "PASS" if main_df[main_df.split == s_].region.nunique() == 6 else "WARN")
ks = {f: ks_2samp(main_df[main_df.split == "train"][f], main_df[main_df.split == "test"][f]).statistic for f in Fy if not f.startswith(("crop_", "prev_"))}
chk(S, "équilibre des distributions train vs test : KS max", max(ks.values()), None, 0.15, note=max(ks, key=ks.get), warn=True)
chk(S, "équilibre des distributions train vs test : KS moyen", float(np.mean(list(ks.values()))), None, 0.08)
ev = main_df.groupby("split").event.apply(lambda s: (s != "none").mean())
add(S, "part d'événements extrêmes par split", ", ".join(f"{k} {v:.0%}" for k, v in ev.items()), "similaire (~30 %)", "PASS" if ev.max() - ev.min() < 0.1 else "WARN")

# =============================================================== 3. MACHINE LEARNING
S = "3 Machine learning"
for m in MODELS:
    mt = meta[m]
    flag(S, f"{m}: séparation train/val/test appliquée (tailles {mt['n_train']}/{mt['n_val']}/{mt['n_test']})", min(mt["n_train"], mt["n_val"], mt["n_test"]) > 0)
    chk(S, f"{m}: parité ONNX vs scikit-learn (écart max)", mt["onnx_parity"], None, 1e-3, "{:.1e}")
    chk(S, f"{m}: stabilité validation croisée par groupes (écart-type R2)", mt["cv_r2_group"]["std"], None, 0.10)
    chk(S, f"{m}: faux positifs hors-domaine sur test", mt["ood_false_positive_test"], None, 0.05)
    for t, d in mt["diagnostics"].items():
        chk(S, f"{m}/{t}: sur-apprentissage (écart val/train relatif)", d["gap_rel"], None, 0.30, warn=True)
        add(S, f"{m}/{t}: R2 train/val/test", f"{d['r2']['tr']:.2f} / {d['r2']['va']:.2f} / {d['r2']['te']:.2f}", "R2 val >= 0.15 (sinon sous-apprentissage)",
            "PASS" if d["r2"]["va"] >= 0.15 else "WARN", "" if d["r2"]["va"] >= 0.15 else "peu prévisible avant la saison : l'intervalle porte l'information")
        chk(S, f"{m}/{t}: couverture de l'intervalle à 80 % sur test", d["coverage_test"], 0.72, 0.88, "{:.2f}", warn=True)
        chk(S, f"{m}/{t}: gain vs prédiction constante", d["skill_vs_baseline"], 0.05, None, "{:.2f}", warn=True)
        lc = d["learning_curve"]; imp = (lc[1]["mae_val"] - lc[2]["mae_val"]) / max(lc[1]["mae_val"], 1e-9)
        add(S, f"{m}/{t}: courbe d'apprentissage (gain val 50%->100 % des données)", f"{imp:.1%}", "< 5 % = données suffisantes", "PASS" if imp < 0.05 else "INFO",
            "" if imp < 0.05 else "plus de données amélioreraient encore le modèle")
    add(S, f"{m}: corrections automatiques proposées", " | ".join(mt["advice"])[:400], "", "INFO")

def feats(df, m): return df[meta[m]["features"]]
test = main_df[main_df.split == "test"].reset_index(drop=True)
rng = np.random.default_rng(1)
for m in ("soil", "crop", "yield"):
    P = Predictor(f"models/{m}"); X = feats(test, m)
    base = P.predict(X)
    Xs = X.copy(); num = [c for c in X.columns if not c.startswith(("crop_", "prev_"))]
    Xs[num] = Xs[num] * rng.normal(1, 0.05, (len(X), len(num)))
    sh = P.predict(Xs); ys = np.mean([meta[m]["target_meta"][t]["y_std"] for t in meta[m]["targets"]])
    chg = np.mean([np.abs(sh[f"{t}_mid"] - base[f"{t}_mid"]).mean() / meta[m]["target_meta"][t]["y_std"] for t in meta[m]["targets"]])
    chk(S, f"{m}: robustesse au bruit +-5 % sur les entrées (variation relative moyenne des sorties)", float(chg), None, 0.10, "{:.3f}")
    Xo = X.copy(); j = num[0]; Xo[j] = Xo[j] + 8 * (X[j].std() + 1e-9)
    flag(S, f"{m}: détection hors-domaine (décalage de 8 écarts-types sur '{j}')", float(P.predict(Xo).ood_flag.mean()) >= 0.95, f"détecté {P.predict(Xo).ood_flag.mean():.0%}")
    chk(S, f"{m}: baisse de confiance sur données hors domaine", float(P.predict(Xo)[f"{meta[m]['targets'][0]}_conf"].mean() / base[f"{meta[m]['targets'][0]}_conf"].mean()), None, 0.6, "{:.2f}")
# généralisation spatiale : Yield AI sans une région
Fy_ = meta["yield"]["features"]
for held in ("bukavu", "lubumbashi"):
    tr_ = main_df[(main_df.split == "train") & (main_df.region != held)]; te_ = main_df[(main_df.split == "test") & (main_df.region == held)]
    _, ci, ms = _fit_heads(tr_[Fy_].to_numpy(np.float32), tr_["yield_t_ha"].to_numpy(), te_[Fy_].to_numpy(np.float32), te_["yield_t_ha"].to_numpy(), 0, CONFIGS[:1])
    r2_out = _r2(te_["yield_t_ha"].to_numpy(), ms[1].predict(te_[Fy_].to_numpy(np.float32)))
    Pf = Predictor("models/yield"); te_in = main_df[(main_df.split == "test") & (main_df.region == held)]
    r2_in = _r2(te_in["yield_t_ha"].to_numpy(), Pf.predict(te_in[Fy_])["yield_t_ha_mid"].to_numpy())
    add(S, f"Yield AI : région '{held}' jamais vue (R2) vs région vue", f"{r2_out:.2f} vs {r2_in:.2f}", "chute attendue", "INFO",
        "mesure de la généralisation spatiale ; si la chute est forte, le système doit signaler « données jamais vues »")
# couverture des scénarios : événements extrêmes et régions (Yield AI, jeu de test)
pp = main_df.drop_duplicates("pair_id"); ntr = pp[pp.split == "train"].event.value_counts()
chk("2 Données", "couverture : paires d'entraînement par type d'événement (minimum, hors « none »)", int(ntr.drop("none").min()), 50, None, "{:.0f}", str(ntr.to_dict()), warn=True)
rn_ = pp.clim_season_rain_mm / pp.rain_norm_season
chk("2 Données", "couverture : saisons très sèches (pluie < 60 % de la normale, %)", 100 * float((rn_ < 0.6).mean()), 5, None, "{:.1f}", "extrême rare : peu d'exemples pour apprendre la sécheresse", warn=True)
chk("2 Données", "couverture : saisons avec au moins 3 jours à 35 °C ou plus (%)", 100 * float((pp.clim_heat_days >= 3).mean()), 5, None, "{:.1f}", "explique l'absence de gain du Climate AI sur la chaleur", warn=True)
Py = Predictor("models/yield"); oy = Py.predict(test[Py.features])
for col in ("event", "region"):
    for k, g in test.groupby(col):
        i = g.index; cov = float(((g.yield_t_ha >= oy.loc[i, "yield_t_ha_lo"]) & (g.yield_t_ha <= oy.loc[i, "yield_t_ha_hi"])).mean())
        r2_ = _r2(g.yield_t_ha.to_numpy(), oy.loc[i, "yield_t_ha_mid"].to_numpy()); mae_ = float(np.abs(g.yield_t_ha - oy.loc[i, "yield_t_ha_mid"]).mean())
        add("3 Machine learning", f"Yield AI par {col} = {k} (n={len(g)}) : couverture 80 % / R2 / MAE t/ha", f"{cov:.0%} / {r2_:.2f} / {mae_:.2f}", "couverture 72-88 %, R2 >= 0.6",
            "PASS" if (0.72 <= cov <= 0.88 and r2_ >= 0.6) or len(g) < 30 else "WARN", "" if (0.72 <= cov <= 0.88 and r2_ >= 0.6) or len(g) < 30 else ("trop sûr de lui : intervalle trop étroit" if cov < 0.72 else ("intervalle trop large (prudent, sans danger)" if cov > 0.88 else "précision faible")))
# calibration des probabilités dérivées (IA climat)
Pc = Predictor("models/climate"); ct = clim_df[clim_df.split == "test"].reset_index(drop=True); oc = Pc.predict(ct[meta["climate"]["features"]])
def reliab(ev, thr_fn, name):
    p = np.array([prob_exceed(thr_fn(i), oc.loc[i, f"{ev}_lo"], oc.loc[i, f"{ev}_mid"], oc.loc[i, f"{ev}_hi"]) for i in range(len(ct))])
    return p
p14 = reliab("dry_spell_days", lambda i: 14, "x"); y14 = (ct.dry_spell_days > 14).to_numpy().astype(float)
p100 = reliab("max3_rain_mm", lambda i: 100, "x"); y100 = (ct.max3_rain_mm > 100).to_numpy().astype(float)
def ece(p, y, nb=5):
    b = np.minimum((p * nb).astype(int), nb - 1); return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any()))
for nm, p, y in (("P(saison sèche >14 j)", p14, y14), ("P(pluie 3 j >100 mm)", p100, y100)):
    brier = float(np.mean((p - y) ** 2)); b0 = float(np.mean((y.mean() - y) ** 2))
    chk(S, f"calibration {nm} : ECE", ece(p, y), None, 0.10, "{:.3f}", f"Brier {brier:.3f} vs climatologie {b0:.3f}", warn=True)
# --- chaîne complète sur les paires de test
def chain(df):
    Ps = {m: Predictor(f"models/{m}") for m in MODELS}; rows = []
    for pid, g in df.groupby("pair_id"):
        g = g.set_index("crop")
        cl = Ps["climate"].predict(g.iloc[[0]][meta["climate"]["features"]])
        so = Ps["soil"].predict(g[meta["soil"]["features"]]); cr = Ps["crop"].predict(g[meta["crop"]["features"]]); yl = Ps["yield"].predict(g[meta["yield"]["features"]])
        Rr = pd.DataFrame(index=g.index)
        for t in ("season_rain_mm", "dry_spell_days", "max3_rain_mm", "heat_days"): Rr[t] = float(cl[f"{t}_mid"].iloc[0])
        for t in ("erosion_t_ha", "d_soc_pct", "d_ph", "n_leach_kg"): Rr[t] = so[f"{t}_mid"]
        for t in ("days_to_harvest", "water_use_mm", "water_stress", "n_uptake_kg"): Rr[t] = cr[f"{t}_mid"]
        Rr["yield_t_ha"] = yl["yield_t_ha_mid"]
        for c in CROPS: Rr[f"crop_{c}"] = g[f"crop_{c}"]
        rc = Ps["recommendation"].predict(Rr)
        cc = float(cl[[f"{t}_conf" for t in ("season_rain_mm", "dry_spell_days", "max3_rain_mm", "heat_days")]].mean(axis=1).iloc[0])
        d = decide(rc, so, cr, yl, cc, False)
        ref = g["score_ref"]
        rows.append(dict(pair=pid, top=d["top"], conf=d["conf"], p_top=d["p_top"], regret=float(ref.max() - ref[d["top"]]),
                         best=ref.idxmax(), ref_best=float(ref.max()), ref_top=float(ref[d["top"]]), region=g.region.iloc[0],
                         cov=float(np.mean([(ref[c] >= rc.loc[c, "score_ref_lo"]) and (ref[c] <= rc.loc[c, "score_ref_hi"]) for c in g.index])),
                         mae=float(np.mean([abs(ref[c] - rc.loc[c, "score_ref_mid"]) for c in g.index])),
                         ood=bool(so.ood_flag.any() or yl.ood_flag.any() or cr.ood_flag.any())))
    return pd.DataFrame(rows)
E = chain(test)
trm = main_df[main_df.split == "train"]
reg_best = trm.groupby(["region", "crop"]).score_ref.mean().groupby("region").idxmax().apply(lambda x: x[1])
tg = test.groupby("pair_id"); 
def regret_of(f): return float(np.mean([g.score_ref.max() - g.set_index("crop").score_ref[f(g)] for _, g in tg]))
r_ai = float(E.regret.mean()); r_rand = float(np.mean([g.score_ref.max() - g.score_ref.mean() for _, g in tg]))
r_maize = regret_of(lambda g: "maize"); r_reg = regret_of(lambda g: reg_best[g.region.iloc[0]])
rel = lambda r: r / float(np.mean([g.score_ref.max() for _, g in tg]))
add(S, "BOUT EN BOUT (300 paires de test) : regret moyen relatif IA / aléatoire / toujours maïs / meilleure culture par région", f"{rel(r_ai):.1%} / {rel(r_rand):.1%} / {rel(r_maize):.1%} / {rel(r_reg):.1%}",
    "IA < toutes les références", "PASS" if r_ai < min(r_rand, r_maize, r_reg) else "WARN", "regret = score de la meilleure culture (vérité du jumeau) - score de la culture recommandée")
chk(S, "bout en bout : la culture recommandée est la meilleure (exactitude top-1)", float((E.top == E.best).mean()), 0.40, None, "{:.2f}", "pas de chance = 20 % ; classes déséquilibrées", warn=True)
chk(S, "bout en bout : recommandation à moins de 10 % du meilleur score", float((E.ref_top >= 0.9 * E.ref_best).mean()), 0.60, None, "{:.2f}", warn=True)
chk(S, "bout en bout : couverture de l'intervalle du score (80 % nominal)", float(E["cov"].mean()), 0.70, 0.90, "{:.2f}", "chaîne réelle : les erreurs des modèles amont se cumulent", warn=True)
E["succ"] = (E.ref_top >= 0.9 * E.ref_best).astype(float); E["q"] = pd.qcut(E.conf, 3, labels=["basse", "moyenne", "haute"], duplicates="drop")
qs = E.groupby("q", observed=True).succ.mean()
add(S, "confiance affichée vs réussite réelle (tercile bas/moyen/haut)", ", ".join(f"{k} {v:.0%}" for k, v in qs.items()), "croissante", "PASS" if qs.is_monotonic_increasing else "WARN",
    f"confiance moyenne {E.conf.mean():.0%} ; réussite moyenne {E.succ.mean():.0%}")
chk(S, "confiance moyenne - réussite moyenne (sur-confiance si > 0)", float(E.conf.mean() - E.succ.mean()), -0.20, 0.10, "{:+.2f}", "la confiance ne doit pas dépasser la réussite observée", warn=True)
chk(S, "confiance maximale affichée (jamais de certitude)", float(E.conf.max()), None, 0.95, "{:.2f}")
E.to_csv("reports/end_to_end_test.csv", index=False)

# =============================================================== 4. JUMEAU NUMÉRIQUE / ADAPTATEURS
S = "4 Jumeau numérique"
if os.path.exists("reports/twin_checks.json"):
    tw = pd.DataFrame(json.load(open("reports/twin_checks.json")))
    for sec, g in tw.groupby("section"): add(S, f"audit du jumeau — {sec}", f"{g.status.isin(['PASS', 'INFO']).sum()}/{len(g)} PASS (INFO inclus)", "aucun FAIL", "PASS" if g.status.isin(["PASS", "INFO"]).all() else "FAIL", "; ".join(g[~g.status.isin(["PASS", "INFO"])].check)[:200])
else:
    add(S, "audit du jumeau", "non exécuté (lancer audit/audit_twin.py)", "", "WARN")
p = subprocess.run([sys.executable, "tests/test_adapters.py"], capture_output=True, text=True, cwd=ROOT)
add(S, "adaptateurs (POWER, GPM, SMAP, MODIS, SoilGrids, FAOSTAT) sur charges fictives", p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-100:], "tous OK", "PASS" if p.returncode == 0 else "FAIL",
    "format documenté, JAMAIS testé contre les services réels")

# =============================================================== 5. VALIDATION SCIENTIFIQUE
S = "5 Validation scientifique"
import tempfile, importlib.util
sp_ = importlib.util.spec_from_file_location("validate", "audit/validate.py"); val = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(val)
txt_real = val.build(con)
ext_lines = [l for l in txt_real.splitlines() if l.startswith("- **") and any(k in l for k in ("GPM", "SMAP", "MODIS"))]
n_real = sum(con.execute(f"SELECT COUNT(*) FROM {t} WHERE source=?", (s_,)).fetchone()[0] for t, s_ in (("weather", "gpm"), ("soil_moisture", "smap"), ("vegetation", "modis")))
flag(S, "aucune validation externe déclarée sans données réelles (GPM/SMAP/MODIS : « NON RÉALISÉE »)", n_real > 0 or all("NON RÉALISÉE" in l for l in ext_lines), f"lignes réelles en base : {n_real}")
with tempfile.TemporaryDirectory() as td:       # TEST DU CODE avec des données FACTICES : sans valeur scientifique
    from core.db import put
    c2 = connect(os.path.join(td, "t.db")); R0 = REGIONS["kananga"]; w = generate_weather(R0, "2010-01-01", 365 * 5, 3); w.insert(0, "region", "kananga")
    put(c2, "weather", w, "nasa_power"); g = w[["region", "date", "PRECTOTCORR"]].copy(); g["PRECTOTCORR"] *= 1.1; put(c2, "weather", g, "gpm")
    d = pd.date_range("2012-01-01", periods=120, freq="3D"); put(c2, "soil_moisture", pd.DataFrame(dict(region="kananga", date=d, value=0.2 + 0.1 * np.random.default_rng(0).random(len(d)), kind="smap_surface")), "smap")
    put(c2, "vegetation", pd.DataFrame(dict(region="kananga", date=pd.date_range("2012-01-01", periods=60, freq="16D"), ndvi=0.6)), "modis")
    t2 = val.build(c2)
flag(S, "code de validation exécutable sur données factices (test du code uniquement)", all("NON RÉALISÉE" not in l for l in t2.splitlines() if l.startswith(("- **GPM**", "- **SMAP**", "- **NASA POWER (journalier)**"))), "ce test ne valide PAS le jumeau")
p = subprocess.run([sys.executable, "audit/validate.py"], capture_output=True, text=True, cwd=ROOT)
add(S, "rapport de validation (reports/validation_report.md)", "généré" if p.returncode == 0 else p.stderr[-200:], "", "PASS" if p.returncode == 0 else "FAIL",
    "contenu : validation interne / calibration / validation externe (externe = NON RÉALISÉE sans données réelles)")

# =============================================================== sortie
df = pd.DataFrame(R); df.to_json("reports/audit_report.json", orient="records", force_ascii=False, indent=1)
cnt = df.status.value_counts().to_dict()
with open("reports/audit_report.md", "w") as f:
    f.write(f"# Rapport d'audit automatique\n\nGénéré par `audit/audit.py`. Bilan : **{cnt.get('PASS', 0)} PASS · {cnt.get('WARN', 0)} WARN · {cnt.get('FAIL', 0)} FAIL · {cnt.get('INFO', 0)} INFO**.\n\n"
            "WARN = point d'attention documenté (pas une erreur bloquante) ; FAIL = à corriger. Les contrôles portent sur la cohérence interne et le comportement des modèles "
            "face à la vérité du jumeau numérique : **ce n'est pas une validation sur données réelles** (voir validation_report.md).\n")
    for sec, g in df.groupby("section"):
        f.write(f"\n## {sec}  ({(g.status == 'PASS').sum()}/{len(g)} PASS)\n\n| contrôle | mesuré | attendu | statut | note |\n|---|---|---|---|---|\n")
        for _, r_ in g.iterrows(): f.write(f"| {r_.check} | {r_.measured} | {r_.expected} | {r_.status} | {r_.note} |\n")
print(df.groupby("section").status.value_counts().unstack(fill_value=0))
for _, r_ in df[df.status.isin(["FAIL", "WARN"])].iterrows(): print(f" [{r_.status}] {r_.section} | {r_.check} : {r_.measured} (attendu {r_.expected}) {r_.note[:100]}")
