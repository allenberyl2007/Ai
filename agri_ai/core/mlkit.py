"""Boîte à outils ML commune aux 5 modèles : entraînement (quantiles 10/50/90 + calibration conforme),
diagnostics sur/sous-apprentissage, détection hors-domaine, export ONNX (écrit à la main : pas de
dépendance à `onnx`), prédiction hors ligne (numpy + onnxruntime)."""
import json, os, struct, time
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)
CONFIGS = [dict(n_estimators=120, max_depth=3, min_samples_leaf=40, learning_rate=0.08, subsample=0.7),
           dict(n_estimators=160, max_depth=4, min_samples_leaf=25, learning_rate=0.08, subsample=0.7),
           dict(n_estimators=100, max_depth=3, min_samples_leaf=100, learning_rate=0.08, subsample=0.6)]
COVER = 0.8   # couverture nominale de l'intervalle [q10-c, q90+c]


# ============================================================ ONNX minimal (protobuf à la main)
def _v(n):
    n = int(n); n = n if n >= 0 else n + (1 << 64)
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        if n: out.append(b | 0x80)
        else: out.append(b); return bytes(out)
def _tag(f, w): return _v((f << 3) | w)
def _ld(f, b): return _tag(f, 2) + _v(len(b)) + b
def _int(f, n): return _tag(f, 0) + _v(n)
def _str(f, s): return _ld(f, s.encode())
def _pints(f, xs): return _ld(f, b"".join(_v(x) for x in xs))
def _pfl(f, xs): return _ld(f, struct.pack("<%df" % len(xs), *xs))
def _a_i(n, v): return _str(1, n) + _int(3, v) + _int(20, 2)
def _a_s(n, v): return _str(1, n) + _ld(4, v.encode()) + _int(20, 3)
def _a_ints(n, xs): return _str(1, n) + _pints(8, xs) + _int(20, 7)
def _a_fl(n, xs): return _str(1, n) + _pfl(7, xs) + _int(20, 6)
def _a_strs(n, xs): return _str(1, n) + b"".join(_ld(9, x.encode()) for x in xs) + _int(20, 8)
def _vinfo(name, last_dim):
    dim_n = _ld(1, _str(2, "N")); dim_l = _ld(1, _int(1, last_dim))
    tensor = _int(1, 1) + _ld(2, dim_n + dim_l)
    return _str(1, name) + _ld(2, _ld(1, tensor))


def export_onnx(models, n_features, path, props=None):
    """models : liste de GradientBoostingRegressor ; sortie Y[:, j] = prédiction du modèle j."""
    nt, ni, nf, nv, nm, ntr, nfa = [], [], [], [], [], [], []
    tt, tn, tid, tw = [], [], [], []
    base, tree_id = [], 0
    for j, m in enumerate(models):
        lr = m.learning_rate
        base.append(float(m.init_.predict(np.zeros((1, n_features)))[0]))
        for k in range(m.estimators_.shape[0]):
            t = m.estimators_[k, 0].tree_
            for node in range(t.node_count):
                nt.append(tree_id); ni.append(node)
                if t.children_left[node] == -1:
                    nf.append(0); nv.append(0.0); nm.append("LEAF"); ntr.append(0); nfa.append(0)
                    tt.append(tree_id); tn.append(node); tid.append(j); tw.append(float(lr * t.value[node, 0, 0]))
                else:
                    nf.append(int(t.feature[node])); nv.append(float(np.float32(t.threshold[node]))); nm.append("BRANCH_LEQ")
                    ntr.append(int(t.children_left[node])); nfa.append(int(t.children_right[node]))
            tree_id += 1
    attrs = [_a_i("n_targets", len(models)), _a_ints("nodes_featureids", nf), _a_strs("nodes_modes", nm),
             _a_ints("nodes_nodeids", ni), _a_ints("nodes_treeids", nt), _a_ints("nodes_truenodeids", ntr),
             _a_ints("nodes_falsenodeids", nfa), _a_fl("nodes_values", nv), _a_ints("target_ids", tid),
             _a_ints("target_nodeids", tn), _a_ints("target_treeids", tt), _a_fl("target_weights", tw),
             _a_fl("base_values", base), _a_s("aggregate_function", "SUM"), _a_s("post_transform", "NONE")]
    node = _str(1, "X") + _str(2, "Y") + _str(3, "trees") + _str(4, "TreeEnsembleRegressor") + _str(7, "ai.onnx.ml") + \
        b"".join(_ld(5, a) for a in attrs)
    graph = _ld(1, node) + _str(2, "agri") + _ld(11, _vinfo("X", n_features)) + _ld(12, _vinfo("Y", len(models)))
    model = _int(1, 8) + _str(2, "agri_ai") + _ld(7, graph) + _ld(8, _str(1, "") + _int(2, 13)) + \
        _ld(8, _str(1, "ai.onnx.ml") + _int(2, 3)) + b"".join(_ld(14, _str(1, k) + _str(2, v)) for k, v in (props or {}).items())
    with open(path, "wb") as f:
        f.write(model)


# ============================================================ entraînement
def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def _r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def _fit_heads(Xtr, ytr, Xva, yva, seed, cfgs=CONFIGS):
    from sklearn.ensemble import GradientBoostingRegressor as G
    best = None
    for ci, cfg in enumerate(cfgs):
        ms = [G(loss="quantile", alpha=a, random_state=seed, max_features=0.8, **cfg).fit(Xtr, ytr) for a in QS]
        v = float(np.mean([pinball(yva, m.predict(Xva), a) for m, a in zip(ms, QS)]))
        if best is None or v < best[0]:
            best = (v, ci, ms)
    return best


def _knn_ref(Xs, rng, n_ref=3000, k=5):
    ref = Xs[rng.choice(len(Xs), min(n_ref, len(Xs)), replace=False)]
    probe = Xs[rng.choice(len(Xs), min(1500, len(Xs)), replace=False)]
    d = _knn_dist(probe, ref, k, loo=True)
    return ref, float(np.quantile(d, 0.99))


def _knn_dist(Q, R, k=5, loo=False):
    d2 = (Q ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None, :] - 2 * Q @ R.T
    d2 = np.maximum(d2, 0)
    kk = k + 1 if loo else k
    part = np.partition(d2, kk - 1, axis=1)[:, :kk]
    part.sort(axis=1)
    return np.sqrt(part[:, -1]) if not loo else np.sqrt(part[:, k])


def train_model(name, df, features, targets, outdir, units=None, seed=0):
    """df doit contenir la colonne `split` (train/val/test) ; aucune autre colonne que features/targets n'est lue."""
    t0 = time.time(); rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    X = {k: d[features].to_numpy(np.float32) for k, d in (("tr", tr), ("va", va), ("te", te))}
    mu, sd = X["tr"].mean(0), X["tr"].std(0) + 1e-9
    allm, meta_t, diag = [], {}, {}
    for t in targets:
        y = {k: d[t].to_numpy(float) for k, d in (("tr", tr), ("va", va), ("te", te))}
        v, ci, ms = _fit_heads(X["tr"], y["tr"], X["va"], y["va"], seed)
        P = {k: np.sort(np.column_stack([m.predict(X[k]) for m in ms]), axis=1) for k in X}
        s = np.maximum(P["va"][:, 0] - y["va"], y["va"] - P["va"][:, 2])
        n = len(s); c = float(np.quantile(s, min(1.0, np.ceil((n + 1) * COVER) / n)))
        cov = lambda k: float(np.mean((y[k] >= P[k][:, 0] - c) & (y[k] <= P[k][:, 2] + c)))
        mae = {k: float(np.mean(np.abs(y[k] - P[k][:, 1]))) for k in X}
        r2 = {k: _r2(y[k], P[k][:, 1]) for k in X}
        # courbe d'apprentissage (tête médiane seule)
        from sklearn.ensemble import GradientBoostingRegressor as G
        lc = []
        for frac in (0.25, 0.5, 1.0):
            idx = rng.choice(len(X["tr"]), int(frac * len(X["tr"])), replace=False)
            m = G(loss="quantile", alpha=0.5, random_state=seed, max_features=0.8, **CONFIGS[ci]).fit(X["tr"][idx], y["tr"][idx])
            lc.append(dict(n=int(len(idx)), mae_train=float(np.mean(np.abs(y["tr"][idx] - m.predict(X["tr"][idx])))),
                           mae_val=float(np.mean(np.abs(y["va"] - m.predict(X["va"]))))))
        # baseline : médiane d'apprentissage
        base_mae = float(np.mean(np.abs(y["te"] - np.median(y["tr"]))))
        meta_t[t] = dict(conformal_c=c, y_std=float(y["tr"].std()), y_median=float(np.median(y["tr"])), config=ci)
        diag[t] = dict(mae=mae, r2=r2, coverage_val=cov("va"), coverage_test=cov("te"), conformal_c=c,
                       gap_rel=float((mae["va"] - mae["tr"]) / max(mae["va"], 1e-9)), learning_curve=lc,
                       baseline_mae_test=base_mae, skill_vs_baseline=float(1 - mae["te"] / max(base_mae, 1e-12)),
                       resid_std_test=float(np.std(y["te"] - P["te"][:, 1])), unseen_ratio=None)
        allm += ms
    # stabilité : validation croisée par groupes (tête médiane, 1ère cible) sur train
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import GradientBoostingRegressor as G
    t = targets[0]; cvr = []
    grp = tr["wseed"].to_numpy() if "wseed" in tr else np.arange(len(tr))
    for a, b in GroupKFold(3).split(X["tr"], groups=grp):
        m = G(loss="quantile", alpha=0.5, random_state=seed, max_features=0.8, **CONFIGS[meta_t[t]["config"]]).fit(X["tr"][a], tr[t].to_numpy()[a])
        cvr.append(_r2(tr[t].to_numpy()[b], m.predict(X["tr"][b])))
    # hors-domaine
    Xs = (X["tr"] - mu) / sd
    ref, thr = _knn_ref(Xs, rng)
    thr = max(thr, float(np.quantile(_knn_dist((X["va"] - mu) / sd, ref, 5), 0.99)))   # seuil aussi calibré sur la validation (nouveaux sites)
    import io, base64
    buf = io.BytesIO()
    np.savez_compressed(buf, mu=mu, sd=sd, ref=ref.astype(np.float32), fmin=X["tr"].min(0), fmax=X["tr"].max(0), thr=np.float32(thr))
    ref_b64 = base64.b64encode(buf.getvalue()).decode()
    onnx_path = os.path.join(outdir, "model.onnx")
    export_onnx(allm, len(features), onnx_path)          # écrit une 1re fois (test de parité), réécrit avec métadonnées ensuite
    # parité ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    Y = sess.run(None, {"X": X["te"]})[0]
    ref_pred = np.column_stack([m.predict(X["te"]) for m in allm])
    parity = float(np.abs(Y - ref_pred).max())
    # taux de faux positifs OOD sur test (données du même domaine)
    d_te = _knn_dist((X["te"] - mu) / sd, ref, 5)
    ood_fp = float(np.mean(d_te > thr))
    for t in targets: diag[t]["onnx_parity_max_abs"] = parity
    meta = dict(name=name, features=features, targets=targets, units=units or {}, quantiles=QS, cover=COVER,
                target_meta=meta_t, ood_threshold=float(thr), ood_false_positive_test=ood_fp, onnx_parity=parity,
                n_train=len(tr), n_val=len(va), n_test=len(te), cv_r2_group=dict(mean=float(np.mean(cvr)), std=float(np.std(cvr)), target=t),
                diagnostics=diag, advice=advise(diag, cvr), seconds=round(time.time() - t0, 1), trained_on=str(df["source"].iloc[0]) if "source" in df else "sim")
    export_onnx(allm, len(features), onnx_path, {"agri_meta": json.dumps(meta, ensure_ascii=False), "agri_ref": ref_b64})
    return meta


def advise(diag, cvr):
    """Corrections proposées automatiquement (diagnostic biais/variance)."""
    out = []
    for t, d in diag.items():
        if d["gap_rel"] > 0.30:
            out.append(f"{t}: variance élevée (écart train/val {d['gap_rel']:.0%}) -> augmenter min_samples_leaf, réduire la profondeur, ajouter des données")
        if d["r2"]["va"] < 0.15 and d["r2"]["tr"] < 0.30:
            out.append(f"{t}: biais élevé (R2 train {d['r2']['tr']:.2f}, val {d['r2']['va']:.2f}) -> ajouter des variables informatives ou de la capacité")
        if d["skill_vs_baseline"] < 0.05:
            out.append(f"{t}: aucun gain sur la médiane constante ({d['skill_vs_baseline']:.0%}) -> variable difficilement prévisible ; la confiance est limitée à l'intervalle")
        if abs(d["coverage_test"] - COVER) > 0.07:
            out.append(f"{t}: couverture de l'intervalle sur test {d['coverage_test']:.0%} (cible {COVER:.0%}) -> recalibrer sur plus de données de validation")
    if np.std(cvr) > 0.10:
        out.append(f"instabilité entre plis de validation croisée (écart-type R2 {np.std(cvr):.2f}) -> plus de données ou modèle plus simple")
    return out or ["aucune correction nécessaire selon les seuils automatiques"]


# ============================================================ prédiction (hors ligne)
def read_meta(d):
    """Métadonnées embarquées dans model.onnx (diagnostics d'entraînement, variables, conformité)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(d, "model.onnx"), providers=["CPUExecutionProvider"])
    return json.loads(sess.get_modelmeta().custom_metadata_map["agri_meta"]), sess


class Predictor:
    def __init__(self, d):
        import io, base64
        self.meta, self.sess = read_meta(d)
        z = np.load(io.BytesIO(base64.b64decode(self.sess.get_modelmeta().custom_metadata_map["agri_ref"])))
        self.mu, self.sd, self.ref, self.thr, self.fmin, self.fmax = z["mu"], z["sd"], z["ref"], float(z["thr"]), z["fmin"], z["fmax"]
        self.features, self.targets = self.meta["features"], self.meta["targets"]

    def ood(self, X):
        d = _knn_dist((X - self.mu) / self.sd, self.ref, 5)
        score = d / self.thr
        span = np.maximum(self.fmax - self.fmin, 1e-9)
        out_of_range = (X < self.fmin - 0.05 * span) | (X > self.fmax + 0.05 * span)
        return score, out_of_range

    def predict(self, df):
        miss = [f for f in self.features if f not in df.columns]
        if miss: raise KeyError(f"variables manquantes : {miss}")
        X = df[self.features].to_numpy(np.float32)
        if np.isnan(X).any(): raise ValueError("NaN dans les variables d'entrée")
        Y = self.sess.run(None, {"X": X})[0]
        score, oor = self.ood(X)
        f_ood = np.clip(1 - (score - 0.7) / 0.8 * 0.8, 0.2, 1.0)
        res = pd.DataFrame(index=df.index)
        for i, t in enumerate(self.targets):
            q = np.sort(Y[:, 3 * i:3 * i + 3], axis=1); tm = self.meta["target_meta"][t]; c = tm["conformal_c"]
            lo, mid, hi = q[:, 0] - c, q[:, 1], q[:, 2] + c
            lo = np.minimum(lo, mid); hi = np.maximum(hi, mid)
            w_rel = (hi - lo) / (np.abs(mid) + 0.25 * tm["y_std"] + 1e-9)
            res[f"{t}_lo"], res[f"{t}_mid"], res[f"{t}_hi"] = lo, mid, hi
            res[f"{t}_unc"] = (hi - lo) / 2
            res[f"{t}_conf"] = np.minimum(0.95, 1 / (1 + w_rel)) * f_ood      # confiance : plus l'intervalle est large, plus elle baisse
        res["ood_score"] = score; res["ood_flag"] = score > 1.0
        res["unseen_features"] = [[self.features[j] for j in np.where(r)[0]] for r in oor]
        return res


def prob_exceed(thr, lo, mid, hi):
    """P(Y > thr) par interpolation de la fonction de répartition (niveaux 0.1/0.5/0.9 ~ lo/mid/hi). Approximation."""
    pts_x = np.array([lo, mid, hi], float); pts_p = np.array([0.1, 0.5, 0.9])
    if hi - lo < 1e-9: return float(thr < mid)
    sl_lo = (0.5 - 0.1) / max(mid - lo, 1e-9); sl_hi = (0.9 - 0.5) / max(hi - mid, 1e-9)
    if thr <= lo: cdf = max(0.01, 0.1 - (lo - thr) * sl_lo * 0.5)
    elif thr >= hi: cdf = min(0.99, 0.9 + (thr - hi) * sl_hi * 0.5)
    else: cdf = float(np.interp(thr, pts_x, pts_p))
    return float(1 - cdf)
