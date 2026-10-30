"""Règle de décision commune (main.py ET audit) : classement, probabilité que le 1er batte le 2e, confiance finale."""
import math
import numpy as np
import pandas as pd

SIM_CAP = 0.85      # plafond de confiance tant qu'aucune donnée réelle n'a été utilisée
REAL_CAP = 0.95     # jamais de certitude


def phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def decide(rc, so, cr, yl, clim_conf, real_used=False):
    """rc/so/cr/yl : sorties des modèles (une ligne par culture candidate, index = nom de culture)."""
    mc = lambda df, ts: df[[f"{t}_conf" for t in ts]].mean(axis=1)
    models = pd.concat([mc(so, ["erosion_t_ha", "d_soc_pct", "d_ph", "n_leach_kg"]).rename("soil"),
                        mc(cr, ["days_to_harvest", "water_use_mm", "water_stress", "n_uptake_kg"]).rename("crop"),
                        yl["yield_t_ha_conf"].rename("yield"), rc["score_ref_conf"].rename("reco")], axis=1)
    overall = np.minimum(models.min(axis=1), clim_conf)
    order = rc["score_ref_mid"].sort_values(ascending=False).index.tolist()
    top, alt = order[0], order[1]
    sig = lambda c: (rc.loc[c, "score_ref_hi"] - rc.loc[c, "score_ref_lo"]) / 2.56
    p_top = phi((rc.loc[top, "score_ref_mid"] - rc.loc[alt, "score_ref_mid"]) / math.sqrt(sig(top) ** 2 + sig(alt) ** 2 + 1e-9))
    cap = REAL_CAP if real_used else SIM_CAP
    return dict(order=order, top=top, alt=alt, p_top=p_top, overall=overall, cap=cap, conf=float(min(cap, overall[top]) * p_top))
