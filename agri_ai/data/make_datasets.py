"""Le jumeau numérique génère les datasets (tables ds_main, ds_climate dans simulation.db).
Séparation STRICTE : les sites (sols) ET les graines météo sont attribués à un seul des jeux train/val/test."""
import sys, os, time, argparse, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from core.regions import REGIONS, SOILS, SEASONS
from core.climate import generate_weather, et0_fao56
from core.twin import DigitalTwin, Site, Management
from core.events import Event, apply_weather_events
from core.features import pre_season_features, season_targets
from core.policy import CROPS, score
from core.db import connect

N_SITES = 360
SEED_RANGE = {"train": (0, 700), "val": (700, 850), "test": (850, 1000)}
PREV = [None, "maize", "bean", "groundnut", "rice_upland"]


def site_split(i):                      # 70/15/15 sur les sites (mêmes proportions dans chaque région : i % 6 = région)
    k = (i // 6) % 20
    return "train" if k < 14 else ("val" if k < 17 else "test")


def make_site(i, rng):
    reg = list(REGIONS)[i % len(REGIONS)]
    base = SOILS[rng.choice(REGIONS[reg].soils)]
    ln = lambda s: float(np.exp(rng.normal(0, s)))
    wp = float(np.clip(base.wp * rng.normal(1, 0.10), 0.03, 0.30)); fc = float(np.clip(base.fc * rng.normal(1, 0.08), wp + 0.04, 0.48))
    sat = float(max(fc + 0.05, min(0.6, base.sat * rng.normal(1, 0.05))))
    soil = dataclasses.replace(base, wp=wp, fc=fc, sat=sat, drain=float(np.clip(base.drain + rng.normal(0, 0.1), 0.15, 0.95)),
                               cn=float(np.clip(base.cn + rng.normal(0, 3), 55, 92)), k_erod=base.k_erod * ln(0.25),
                               soc=float(np.clip(base.soc * ln(0.35), 0.3, 8.0)), ph=float(np.clip(base.ph + rng.normal(0, 0.4), 4.0, 7.2)),
                               p_bray=float(np.clip(base.p_bray * ln(0.5), 1.0, 40)), k_exch=float(np.clip(base.k_exch * ln(0.4), 0.03, 1.2)),
                               cec=float(np.clip(base.cec * ln(0.3), 2, 40)), p_fix=float(np.clip(base.p_fix + rng.normal(0, 0.1), 0.1, 0.9)))
    slope = float(np.clip(np.exp(rng.normal(np.log(6), 0.7)), 1, 35))
    return dict(site_id=i, region=reg, soil_base=base.name, soil=soil, slope_pct=slope)


def make_mgmt(rng):
    pick = lambda p, lo, hi: 0.0 if rng.random() < p else float(rng.uniform(lo, hi))
    return dict(n_kg=pick(0.5, 20, 120), p_kg=pick(0.65, 10, 30), k_kg=pick(0.75, 20, 50), lime_t=pick(0.85, 0.5, 2.0),
                weeding=float(rng.uniform(0.3, 0.9)), residue_return=float(rng.uniform(0, 1)),
                pest_control=pick(0.7, 0.3, 0.9), seed_quality=float(rng.uniform(0.2, 0.9)))


def make_events(rng, idx):
    if rng.random() > 0.30: return ()
    day = idx + int(rng.integers(0, 90))
    k = rng.choice(["drought", "dry_spell", "flood", "heatwave", "pest", "disease"])
    d = {"drought": (20, 60, 0.5, 0.9), "dry_spell": (10, 25, 1, 1), "flood": (2, 4, 60, 150), "heatwave": (5, 15, 3, 8),
         "pest": (10, 25, 0.1, 0.5), "disease": (30, 80, 0.3, 1.0)}[k]
    return (Event(k, day, int(rng.integers(d[0], d[1] + 1)), float(rng.uniform(d[2], d[3]))),)


def window_idx(w, region, season):
    wins = SEASONS[region]; sd, _ = wins[min(season, len(wins) - 1)]
    dt = pd.DatetimeIndex(w["date"])
    return int(np.where((dt.year == 2001) & (dt.dayofyear == sd))[0][0])


def build(n_pairs, n_climate, seed=0):
    rng = np.random.default_rng(seed)
    sites = [make_site(i, rng) for i in range(N_SITES)]
    by_split = {s: [x for x in sites if site_split(x["site_id"]) == s] for s in SEED_RANGE}
    rows, crow = [], []
    pid = 0; t0 = time.time()
    for split, n in n_pairs.items():
        for _ in range(n):
            st = by_split[split][rng.integers(len(by_split[split]))]
            wseed = int(rng.integers(*SEED_RANGE[split]))
            reg = st["region"]; season = int(rng.integers(0, len(SEASONS[reg])))
            w = generate_weather(REGIONS[reg], "2000-01-01", 1095, wseed)
            idx = window_idx(w, reg, season)
            ev = make_events(rng, idx); mg = make_mgmt(rng); prev = PREV[int(rng.integers(len(PREV)))]
            pre = pre_season_features(w, reg, idx)
            clim = season_targets(apply_weather_events(w, ev), reg, idx)
            soil = st["soil"]
            base = dict(pair_id=pid, split=split, site_id=st["site_id"], wseed=wseed, region=reg, season=season, source="sim",
                        event=(ev[0].kind if ev else "none"),
                        soc_pct=soil.soc, ph=soil.ph, p_bray=soil.p_bray, k_exch=soil.k_exch, cec=soil.cec, wp=soil.wp, fc=soil.fc,
                        sat=soil.sat, drain=soil.drain, cn=soil.cn, k_erod=soil.k_erod, slope_pct=st["slope_pct"],
                        **mg, prev_legume=float(prev in ("bean", "groundnut")), prev_none=float(prev is None),
                        **pre, **{"clim_" + k: v for k, v in clim.items()})
            for crop in CROPS:
                base_c = dict(base, prev_same=float(prev == crop), **{f"crop_{c}": float(c == crop) for c in CROPS})
                plan = ([Management(prev, 0, season, residue_return=mg["residue_return"])] if prev else []) + \
                       [Management(crop, 1, season, **mg)]
                o = DigitalTwin(Site(reg, soil.name, st["slope_pct"], soil_obj=soil)).run(w, plan, ev, seed=wseed + 7919 * pid)
                d = o["seasons"]
                d = d[(d.get("skipped", 0) == 0) & (d.year == 1) & (d.crop == crop)] if len(d) else d
                if len(d) == 0: continue
                r = d.iloc[0]
                row = dict(base_c, crop=crop, yield_t_ha=r.yield_t_ha, days_to_harvest=float(r.days),
                           water_use_mm=r.transp_mm + r.evap_mm, water_stress=1 - r.ks_mean, n_uptake_kg=r.N_uptake,
                           erosion_t_ha=r.erosion_t_ha, d_soc_pct=r.soc_end_pct - r.soc_start_pct, d_ph=r.ph_end - r.ph_start,
                           n_leach_kg=r.N_leach)
                row["score_ref"] = score(crop, row["yield_t_ha"], row["days_to_harvest"], row["erosion_t_ha"], row["d_soc_pct"], row["water_stress"])
                rows.append(row)
            pid += 1
            if pid % 200 == 0: print(f"  paires {pid}  {time.time() - t0:.0f}s", flush=True)
    # dataset climat : météo seule (plus de lignes, pas de culture)
    for split, n in n_climate.items():
        for _ in range(n):
            reg = list(REGIONS)[rng.integers(len(REGIONS))]; season = int(rng.integers(0, len(SEASONS[reg])))
            wseed = int(rng.integers(*SEED_RANGE[split]))
            w = generate_weather(REGIONS[reg], "2000-01-01", 1095, wseed)
            idx = window_idx(w, reg, season); ev = make_events(rng, idx)
            crow.append(dict(split=split, wseed=wseed, region=reg, season=season, source="sim", event=(ev[0].kind if ev else "none"),
                             **pre_season_features(w, reg, idx), **season_targets(apply_weather_events(w, ev), reg, idx)))
    return pd.DataFrame(rows), pd.DataFrame(crow)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1400); ap.add_argument("--val", type=int, default=300); ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--climate", type=int, default=6000)
    a = ap.parse_args()
    t = time.time()
    nc = {"train": int(a.climate * .7), "val": int(a.climate * .15), "test": int(a.climate * .15)}
    main_df, clim_df = build({"train": a.train, "val": a.val, "test": a.test}, nc)
    con = connect()
    main_df.to_sql("ds_main", con, if_exists="replace", index=False); clim_df.to_sql("ds_climate", con, if_exists="replace", index=False)
    print("ds_main", main_df.shape, "ds_climate", clim_df.shape, f"{time.time() - t:.0f}s")
