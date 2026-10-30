"""Cœur du jumeau numérique : bilan hydrique 2 couches + cycles N/P/K/C + croissance culturale
(RUE + stress) + érosion + biotique + rotation. Pas de temps : 1 jour. Entrée météo = schéma canonique."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence
import math
import numpy as np
import pandas as pd

from .climate import et0_fao56
from .schema import validate_weather
from .crops import CROPS, Crop
from .regions import REGIONS, SOILS, SEASONS, Z1, Z2, TOPSOIL_MASS_KG, SoilType
from .events import Event, biotic_arrays, soil_fertility_loss, apply_weather_events

K_PER_CMOL = 1525.0          # kg K/ha par cmol/kg (0-30 cm, DA 1.3)
P_PER_PPM = TOPSOIL_MASS_KG / 1e6   # kg P/ha par ppm


@dataclass
class Site:
    region: str
    soil: str
    slope_pct: float = 5.0
    slope_len_m: float = 50.0
    soil_obj: Optional[SoilType] = None   # sol propre au site (SoilGrids / perturbation) ; sinon archétype


@dataclass
class Management:
    crop: str
    year: int = 0
    season: int = 0
    n_kg: float = 0.0
    p_kg: float = 0.0
    k_kg: float = 0.0
    lime_t: float = 0.0
    residue_return: float = 0.3
    pest_control: float = 0.0
    seed_quality: float = 0.5
    sow_offset: int = 0          # décalage volontaire du semis (jours) — pratique/retard paysan
    weeding: float = 0.6         # qualité du désherbage 0-1 (concurrence des adventices : écart station/exploitation)


def gdd_fn(t, tb, topt, tmax):
    if t <= tb or t >= tmax:
        return 0.0
    if t <= topt:
        return t - tb
    return (topt - tb) * (tmax - t) / (tmax - topt)


def cc_potential(phi, ccmax):
    """Couverture potentielle en fonction de la phénologie (0..1+)."""
    if phi < 0.45:
        x = phi / 0.45
        return 0.02 + (ccmax - 0.02) * (3 * x * x - 2 * x ** 3)
    if phi < 0.75:
        return ccmax
    x = min(1.0, (phi - 0.75) / 0.25)
    return ccmax * (1 - 0.7 * x)


def kcb_stage(c: Crop, phi):
    if phi < 0.15: return c.kcb_ini
    if phi < 0.45: return c.kcb_ini + (c.kcb_mid - c.kcb_ini) * (phi - 0.15) / 0.30
    if phi < 0.75: return c.kcb_mid
    return c.kcb_mid + (c.kcb_end - c.kcb_mid) * min(1.0, (phi - 0.75) / 0.25)


class DigitalTwin:
    def __init__(self, site: Site):
        self.site = site
        self.region = REGIONS[site.region]
        self.soil = site.soil_obj or SOILS[site.soil]

    # ------------------------------------------------------------------ état initial
    def _init_state(self, fert_loss: float, rng):
        s = self.soil
        k = 1.0 - fert_loss
        soc_kg = s.soc / 100.0 * TOPSOIL_MASS_KG * k
        st = dict(
            W1=(s.wp + 0.6 * (s.fc - s.wp)) * Z1 * 1000, W2=(s.wp + 0.6 * (s.fc - s.wp)) * Z2 * 1000, pond=0.0,
            soc=soc_kg, n_org=soc_kg / s.c_n, n_min=(12.0 + 6.0 * s.soc) * k,
            p_lab=s.p_bray * P_PER_PPM * k, p_stab=15.0 * s.p_bray * P_PER_PPM * k,
            k_ex=s.k_exch * K_PER_CMOL * k, ph=s.ph - 0.6 * fert_loss,
            res_dm=0.0, res_n=0.0, res_p=0.0, res_k=0.0, res_fn=0.0,
            inoc={c.family: 0.10 for c in CROPS.values()},
        )
        st["p_base"] = st["p_lab"]
        return st

    # ------------------------------------------------------------------ simulation
    def run(self, weather: pd.DataFrame, plan: Sequence[Management], events: Sequence[Event] = (),
            seed: int = 0, trace: bool = False) -> dict:
        rng = np.random.default_rng(seed)
        r, s = self.region, self.soil
        qc = validate_weather(weather)
        if not qc["ok"]:
            raise ValueError(f"météo invalide (réparer/filtrer avant simulation) : {qc}")
        weather = apply_weather_events(weather, events)
        n = len(weather)
        et0 = et0_fao56(weather, r.lat, r.elev)
        rain = weather["PRECTOTCORR"].to_numpy(float).copy()
        tmean = weather["T2M"].to_numpy(float); tmax = weather["T2M_MAX"].to_numpy(float)
        rs = weather["ALLSKY_SFC_SW_DWN"].to_numpy(float); rh = weather["RH2M"].to_numpy(float)
        dates = pd.DatetimeIndex(weather["date"]); doy = dates.dayofyear.to_numpy(); yr = dates.year.to_numpy()
        y0 = int(yr[0])
        pest_ev, dis_mult = biotic_arrays(events, n)
        fert_loss = soil_fertility_loss(events)
        st = self._init_state(fert_loss, rng)

        # constantes sol (mm)
        WP1, FC1, SAT1 = s.wp * Z1 * 1000, s.fc * Z1 * 1000, s.sat * Z1 * 1000
        WP2, FC2, SAT2 = s.wp * Z2 * 1000, s.fc * Z2 * 1000, s.sat * Z2 * 1000
        AD1 = 0.5 * WP1
        S_cn = lambda cn: 25.4 * (1000.0 / cn - 10.0)
        cn_base = s.cn + max(0.0, (self.site.slope_pct - 5.0)) * 0.3
        sl = self.site.slope_pct
        LS = (self.site.slope_len_m / 22.13) ** 0.5 * (0.065 + 0.045 * sl + 0.0065 * sl * sl)
        pond_loss = 0.05 if self.soil.name == "fluvisol_loam" else (0.25 if sl > 2 else 0.10)

        # calendrier : jours de début de fenêtre par plan
        wins = SEASONS[self.site.region]
        queue = []
        for m in plan:
            sd, wd = wins[min(m.season, len(wins) - 1)]
            idx = np.where((yr == y0 + m.year) & (doy == sd))[0]
            if len(idx) == 0:
                continue
            queue.append((int(idx[0]) + m.sow_offset, wd, m))
        queue.sort(key=lambda q: q[0])
        qi = 0

        # cumulateurs globaux de bilan
        T = dict(P=0., Q=0., E=0., Tr=0., D=0., dS0=st["W1"] + st["W2"] + st["pond"],
                 N_in=0., N_fix=0., N_leach=0., N_den=0., N_ero=0., N_rem=0., N_atm=0.,
                 P_in=0., P_ero=0., P_rem=0., K_in=0., K_leach=0., K_ero=0., K_rem=0., C_out=0., eros=0.,
                 max_w_res=0.0)
        stock0 = dict(N=st["n_min"] + st["n_org"], P=st["p_lab"] + st["p_stab"], K=st["k_ex"], C=st["soc"])

        seasons: List[dict] = []
        tr = [] if trace else None
        crop: Optional[Crop] = None
        cs: Dict = {}

        for i in range(n):
            P = rain[i]; et = et0[i]; tm = tmean[i]
            T["P"] += P
            # ---- semis ----
            if crop is None and qi < len(queue):
                ws, wd, m = queue[qi]
                if i >= ws:
                    rain3 = rain[max(0, i - 2):i + 1].sum()
                    ready = (rain3 >= 20.0 and st["W1"] > WP1 + 0.5 * (FC1 - WP1)) or (i >= ws + wd) or m.sow_offset > 0
                    if i > ws + wd + 60:
                        qi += 1; seasons.append(dict(crop=m.crop, year=m.year, season=m.season, skipped=1))
                    elif ready:
                        crop = CROPS[m.crop]; qi += 1
                        cs = dict(m=m, sow=i, phi=0.0, cc=0.02, B=0.0, ks_sum=0.0, ks_n=0, ks_flow=0.0, n_flow=0,
                                  heat=0.0, S=0.0, wl=0.0, surv=1.0, pest_dmg=0.0, upN=0.0, upP=0.0, upK=0.0, fixN=0.0,
                                  Tr=0.0, E=0.0, Q=0.0, D=0.0, rain=0.0, eros=0.0, nleach=0.0,
                                  soc0=st["soc"], ph0=st["ph"], nmin0=st["n_min"], plabile0=st["p_lab"],
                                  topdress=i + 30, fertN=m.n_kg, days=0, ks_min=1.0)
                        # fertilisation de fond
                        n0 = 0.4 * m.n_kg if m.n_kg > 0 else 0.0
                        st["n_min"] += n0; T["N_in"] += n0
                        st["p_lab"] += m.p_kg * (1 - s.p_fix); st["p_stab"] += m.p_kg * s.p_fix; T["P_in"] += m.p_kg
                        st["k_ex"] += m.k_kg; T["K_in"] += m.k_kg
                        if m.lime_t > 0:
                            st["ph"] = min(7.2, st["ph"] + m.lime_t * 0.35 / (1 + s.cec / 10.0))
                        # épisode ravageur aléatoire (si non imposé)
                        pest_p = crop.pest_prob * (1 - 0.8 * m.pest_control)
                        if crop.name == "maize" and rain[i:i + 40].sum() < 40:
                            pest_p *= 1.5
                        if rng.random() < pest_p and pest_ev[i:i + 120].max() == 0:
                            a = i + int(rng.integers(10, 60)); d = int(rng.integers(10, 25)); it = rng.uniform(0.1, 0.5)
                            pest_ev[a:a + d] = 1 - (1 - it) ** (1.0 / d)
            if crop is not None and i == cs["topdress"] and cs["fertN"] > 0:
                n1 = 0.6 * cs["fertN"]; st["n_min"] += n1; T["N_in"] += n1

            cc = cs["cc"] if crop else 0.0

            # ---- eau : ruissellement ----
            sat_frac = (st["W1"] - WP1) / max(FC1 - WP1, 1e-6)
            cn = min(95.0, max(40.0, cn_base * (0.92 + 0.12 * min(1.3, max(0.0, sat_frac))) - 8.0 * cc))
            S_ = S_cn(cn); Ia = 0.1 * S_
            Q = (P - Ia) ** 2 / (P - Ia + S_) if P > Ia else 0.0
            # infiltration de la mare
            room = max(0.0, SAT1 - st["W1"]); inf = min(st["pond"], 0.5 * room)
            st["W1"] += inf; st["pond"] -= inf
            st["W1"] += P - Q
            dr1 = max(0.0, st["W1"] - FC1) * s.drain
            st["W1"] -= dr1; st["W2"] += dr1
            if st["W1"] > SAT1:
                st["pond"] += st["W1"] - SAT1; st["W1"] = SAT1
            pq = st["pond"] * pond_loss; st["pond"] -= pq; Q_tot = Q + pq
            dd = max(0.0, st["W2"] - FC2) * s.drain * 0.8
            st["W2"] -= dd
            if st["W2"] > SAT2:
                dd += st["W2"] - SAT2; st["W2"] = SAT2

            # ---- évaporation sol ----
            kr = min(1.0, max(0.0, (st["W1"] - AD1) / (FC1 - AD1)))
            res_cover = min(0.6, st["res_dm"] / 6000.0)
            Es = min(et * (1 - cc) * (1 - 0.5 * res_cover) * 0.9 * kr, max(0.0, st["W1"] - AD1) * 0.5)
            st["W1"] -= Es

            # ---- résidus & minéralisation ----
            fTm = min(2.5, max(0.1, 2.0 ** ((tm - 25.0) / 10.0)))
            if st["W1"] <= FC1:
                fWm = min(1.0, max(0.1, 0.1 + 0.9 * (st["W1"] - WP1) / (0.6 * (FC1 - WP1))))
            else:
                fWm = 1.0 - 0.5 * min(1.0, (st["W1"] - FC1) / max(SAT1 - FC1, 1e-6))
            kmin = fTm * fWm
            dc = 0.03 / 365.0 * st["soc"] * kmin               # respiration SOC
            st["soc"] -= dc; T["C_out"] += dc
            nmn = 0.03 / 365.0 * st["n_org"] * kmin
            st["n_org"] -= nmn; st["n_min"] += nmn
            if st["res_dm"] > 1e-6:
                kd = 0.02 * kmin
                dm_d = kd * st["res_dm"]; hum_c = 0.15 * 0.40 * dm_d
                dn = kd * st["res_n"]; dp_ = kd * st["res_p"]; dk_ = kd * st["res_k"]
                hum_n = min(dn, hum_c / s.c_n)
                net = CROPS[st["res_crop"]].res_net_n if "res_crop" in st else 0.0
                rem = dn - hum_n
                st["n_min"] += net * rem; st["n_org"] += hum_n + (1 - net) * rem
                st["soc"] += hum_c; T["C_out"] += 0.40 * dm_d - hum_c
                st["p_lab"] += dp_; st["k_ex"] += dk_
                st["res_dm"] -= dm_d; st["res_n"] -= dn; st["res_p"] -= dp_; st["res_k"] -= dk_
            # dépôt atmosphérique
            st["n_min"] += 0.012; T["N_atm"] += 0.012
            # P : équilibre labile <-> stable
            flux = 0.008 * (st["p_base"] - st["p_lab"])
            flux = min(flux, st["p_stab"]) if flux > 0 else flux
            st["p_lab"] += flux; st["p_stab"] -= flux
            # ---- culture ----
            Ta = 0.0; Ks = 1.0; Kw = 1.0; f_nut = 1.0; dB = 0.0; s_t = 1.0
            sat1 = st["W1"] / SAT1
            if crop is not None:
                c = crop; m = cs["m"]
                cs["days"] += 1; cs["rain"] += P
                g = gdd_fn(tm, c.tbase, c.topt, c.tmax)
                cs["phi"] += g / c.gdd_mat
                phi = cs["phi"]
                zr = 0.2 + (c.root_max - 0.2) * min(1.0, phi / 0.5)
                f1 = min(zr, Z1) / Z1; f2 = max(0.0, min(zr - Z1, Z2)) / Z2
                av1 = max(0.0, st["W1"] - WP1) * f1; av2 = max(0.0, st["W2"] - WP2) * f2
                taw = (FC1 - WP1) * f1 + (FC2 - WP2) * f2
                dep = taw - (av1 + av2)
                p_adj = min(0.8, max(0.1, c.p_dep + 0.04 * (5 - et)))
                raw = p_adj * taw
                Ks = 1.0 if dep <= raw else max(0.0, (taw - dep) / max(1e-6, (1 - p_adj) * taw))
                # engorgement
                if sat1 > 0.98 or st["pond"] > 5.0:
                    cs["wl"] += 1.0
                else:
                    cs["wl"] = max(0.0, cs["wl"] - 0.5)
                Kw = 1.0 - c.wl_sens * (1 - math.exp(-0.3 * cs["wl"]))
                if cs["wl"] > c.wl_tol_days:
                    cs["surv"] *= 0.9; cs["cc"] *= 0.9
                    cc = cs["cc"]
                cc_pot = cc_potential(phi, c.cc_max)
                kcb = kcb_stage(c, phi)
                Tp = et * kcb * min(1.0, cc / max(cc_pot, 0.02)) if cc_pot > 0.02 else et * kcb * 0.1
                Ta = min(Ks * Kw * Tp, av1 + av2)
                if av1 + av2 > 1e-9 and Ta > 0:
                    st["W1"] -= Ta * av1 / (av1 + av2); st["W2"] -= Ta * av2 / (av1 + av2)
                water_f = Ta / Tp if Tp > 0.05 else 1.0
                # pH / Al
                f_ph = 1.0 if st["ph"] >= c.ph_crit else max(0.3, 1 - c.ph_sens * (c.ph_crit - st["ph"]) / 1.5)
                # disponibilité P selon pH
                ph = st["ph"]
                phP = max(0.3, 1 - 0.30 * (6.3 - ph)) if ph < 6.3 else (max(0.3, 1 - 0.30 * (ph - 7.2)) if ph > 7.2 else 1.0)
                # maladie
                fT = 1.0 if c.dis_T[0] <= tm <= c.dis_T[1] else 0.2
                fRH = min(1.0, max(0.0, (rh[i] - (c.dis_rh - 15)) / 15.0))
                wet_f = 1.0 if P > 1.0 else 0.5
                inoc = st["inoc"][c.family]
                dS = c.dis_susc * (inoc + 0.05 + (1 - m.seed_quality) * 0.10) * fT * fRH * wet_f * cc * 0.02 * dis_mult[i] * (1 - cs["S"])
                cs["S"] += dS
                # croissance
                fTr = gdd_fn(tm, c.tbase, c.topt, c.tmax) / (c.topt - c.tbase)
                dBp = c.rue * rs[i] * cc * 10.0
                pest = pest_ev[i]
                dB_lim = dBp * fTr * water_f * f_ph * (1 - 0.25 * cs["S"]) * (1 - 0.45 * (1 - m.weeding))
                cN, cP, cK = c.conc
                demN = dB_lim * cN
                fix_pot = 0.0
                if c.fix_frac > 0:
                    fix_pot = demN * c.fix_frac * max(0.0, 1 - st["n_min"] / 120.0) * min(Ks, Kw)
                dem_soilN = demN - fix_pot
                capN = st["n_min"] * 0.50; capP = st["p_lab"] * 0.04 * phP; capK = st["k_ex"] * 0.06
                fN = 1.0 if dem_soilN < 1e-9 else min(1.0, capN / dem_soilN)
                fP = 1.0 if dB_lim * cP < 1e-9 else min(1.0, capP / (dB_lim * cP))
                fK = 1.0 if dB_lim * cK < 1e-9 else min(1.0, capK / (dB_lim * cK))
                f_nut = min(fN, fP, fK)
                dB = dB_lim * f_nut
                uN = dem_soilN * f_nut; fx = fix_pot * f_nut; uP = dB * cP; uK = dB * cK
                st["n_min"] -= uN; st["p_lab"] -= uP; st["k_ex"] -= uK
                cs["upN"] += uN + fx; cs["upP"] += uP; cs["upK"] += uK; cs["fixN"] += fx
                T["N_fix"] += fx
                cs["B"] += dB
                s_t = min(water_f, f_nut, f_ph)
                # dynamique de couverture
                cc_pot_prev = cc_potential(phi - g / c.gdd_mat, c.cc_max)
                if phi < 0.45:
                    cs["cc"] = min(c.cc_max, cs["cc"] + max(0.0, cc_pot - cc_pot_prev) * (0.3 + 0.7 * s_t))
                else:
                    cs["cc"] = max(0.0, cs["cc"] + (cc_pot - cc_pot_prev) - 0.02 * (1 - s_t) * cs["cc"])
                cs["Tr"] += Ta; cs["ks_sum"] += min(Ks, Kw); cs["ks_n"] += 1; cs["ks_min"] = min(cs["ks_min"], Ks)
                if 0.45 <= phi <= 0.65:
                    cs["ks_flow"] += min(Ks, Kw); cs["n_flow"] += 1
                    if tmax[i] > c.heat_thr:
                        cs["heat"] += (tmax[i] - c.heat_thr) * 0.02
                if pest > 0:
                    cs["cc"] *= (1 - pest); cs["pest_dmg"] += pest
            T["Tr"] += Ta

            # pertes N/K par lessivage, dénitrification
            fl = 1 - math.exp(-dr1 / FC1) if dr1 > 0 else 0.0
            nl = st["n_min"] * fl; st["n_min"] -= nl; T["N_leach"] += nl
            kl = st["k_ex"] * 0.15 * fl / (1 + s.cec / 10.0); st["k_ex"] -= kl; T["K_leach"] += kl
            if sat1 > 0.95:
                nd = st["n_min"] * 0.02; st["n_min"] -= nd; T["N_den"] += nd
            # acidification
            st["ph"] -= 5.0e-5 * dr1 / (1 + s.cec / 10.0) * min(1.0, max(0.2, (st["ph"] - 3.8) / 1.5))
            if crop is not None and i == cs["sow"] and cs["m"].n_kg > 0:
                st["ph"] -= 0.0007 * cs["m"].n_kg / (1 + s.cec / 10.0)

            # ---- érosion ----
            if P > 5.0 and Q_tot > 0.0:
                cc_eff = 1 - (1 - cc) * (1 - res_cover)
                Cf = 0.01 + 0.99 * (1 - cc_eff) ** 2
                A = 0.35 * P ** 1.6 * s.k_erod * LS * Cf * min(1.0, 2 * Q_tot / max(P, 1.0))
                A = min(A, 150.0)
                fr = min(0.05, A * 1000.0 / TOPSOIL_MASS_KG * 1.5)
                for key, tag in (("soc", "C_out2"), ("n_org", "N_ero"), ("n_min", "N_ero"), ("p_lab", "P_ero"),
                                 ("p_stab", "P_ero"), ("k_ex", "K_ero")):
                    loss = st[key] * fr; st[key] -= loss
                    if tag != "C_out2": T[tag] += loss
                    else: T["C_out"] += loss
                T["eros"] += A
                if crop is not None: cs["eros"] += A
            T["Q"] += Q_tot; T["E"] += Es; T["D"] += dd
            if crop is not None:
                cs["E"] += Es; cs["Q"] += Q_tot; cs["D"] += dd
                cs["nleach"] += nl
            for fam in st["inoc"]:
                st["inoc"][fam] *= 0.9974

            # ---- récolte ----
            if crop is not None and (cs["phi"] >= 1.0 or cs["days"] >= 420):
                seasons.append(self._harvest(crop, cs, st, i, dates, T, rain))
                crop = None; cs = {}

            if tr is not None:
                tr.append((i, P, et, Q_tot, Es, Ta, st["W1"], st["W2"], st["n_min"], st["p_lab"], st["k_ex"], st["ph"],
                           cs.get("cc", 0.0) if crop else 0.0, cs.get("B", 0.0) if crop else 0.0, Ks, Kw, f_nut))
            # résidu bilan hydrique journalier
            # (P = Q_tot + Es + Ta + dd + dS, tolérance numérique)
            ds = (st["W1"] + st["W2"] + st["pond"])
            if i == 0:
                prev_s = T["dS0"]
            res = P - (Q_tot + Es + Ta + dd) - (ds - prev_s)
            T["max_w_res"] = max(T["max_w_res"], abs(res))
            prev_s = ds

        # ---- bilans de fermeture ----
        plant_n = (cs["upN"] if crop else 0.0); plant_p = (cs["upP"] if crop else 0.0); plant_k = (cs["upK"] if crop else 0.0)
        dS = (st["W1"] + st["W2"] + st["pond"]) - T["dS0"]
        stockN = st["n_min"] + st["n_org"] + st["res_n"] + plant_n
        stockP = st["p_lab"] + st["p_stab"] + st["res_p"] + plant_p
        stockK = st["k_ex"] + st["res_k"] + plant_k
        bal = dict(
            water_resid_mm=T["P"] - (T["Q"] + T["E"] + T["Tr"] + T["D"]) - dS,
            water_max_daily_resid=T["max_w_res"],
            N_resid=(stockN - stock0["N"]) - (T["N_in"] + T["N_fix"] + T["N_atm"] - T["N_leach"] - T["N_den"] - T["N_ero"] - T["N_rem"]),
            P_resid=(stockP - stock0["P"]) - (T["P_in"] - T["P_ero"] - T["P_rem"]),
            K_resid=(stockK - stock0["K"]) - (T["K_in"] - T["K_leach"] - T["K_ero"] - T["K_rem"]),
            rain_total=T["P"], eros_total_t_ha=T["eros"],
        )
        out = dict(seasons=pd.DataFrame(seasons), balance=bal, totals=T, final=st)
        if trace:
            out["trace"] = pd.DataFrame(tr, columns=["i", "rain", "et0", "runoff", "Es", "Ta", "W1", "W2", "N_min", "P_lab",
                                                      "K_ex", "pH", "CC", "B", "Ks", "Kw", "f_nut"])
        return out

    # ------------------------------------------------------------------ récolte
    def _harvest(self, c: Crop, cs, st, i, dates, T, rain):
        m = cs["m"]
        B = cs["B"]
        ks_flow = cs["ks_flow"] / cs["n_flow"] if cs["n_flow"] else 1.0
        ky_pen = min(0.9, c.ky_flow * (1.0 - ks_flow))
        heat_f = 1.0 - min(0.8, cs["heat"])
        dis_loss = c.dis_maxloss * (cs["S"] ** 0.8)
        hi_eff = c.hi * (1 - ky_pen) * heat_f * (1 - dis_loss) * cs["surv"]
        ydm = B * hi_eff
        comm = ydm / 1000.0 / (1.0 - c.moisture)
        upN, upP, upK = cs["upN"], cs["upP"], cs["upK"]
        hN, hP, hK = c.hi_nut
        hv = ydm / max(B * c.hi, 1e-9) if B > 0 else 0.0           # part de l'organe réellement formé
        hv = min(1.0, hv)
        rem_n, rem_p, rem_k = upN * hN * hv, upP * hP * hv, upK * hK * hv
        # le reste de la plante (paille + racines + organe perdu) va au pool résidus (racines : 100 %, paille : residue_return)
        left_n, left_p, left_k = upN - rem_n, upP - rem_p, upK - rem_k
        ret = 0.15 + 0.85 * m.residue_return
        T["N_rem"] += rem_n + left_n * (1 - ret); T["P_rem"] += rem_p + left_p * (1 - ret); T["K_rem"] += rem_k + left_k * (1 - ret)
        res_dm = (B - ydm) * ret
        st["res_dm"] += res_dm; st["res_n"] += left_n * ret; st["res_p"] += left_p * ret; st["res_k"] += left_k * ret
        st["res_crop"] = c.name
        # inoculum : carry-over
        st["inoc"][c.family] = min(1.0, st["inoc"][c.family] * 0.6 + 0.5 * cs["S"])
        sow = cs["sow"]
        return dict(
            crop=c.name, year=m.year, season=m.season, skipped=0, sow_date=dates[sow].date(), harvest_date=dates[i].date(),
            days=cs["days"], yield_t_ha=comm, biomass_t_ha=B / 1000.0, hi_eff=hi_eff,
            rain_mm=cs["rain"], transp_mm=cs["Tr"], evap_mm=cs["E"], runoff_mm=cs["Q"], drain_mm=cs["D"],
            ks_mean=cs["ks_sum"] / max(cs["ks_n"], 1), ks_min=cs["ks_min"], waterlog_days=cs["wl"], heat_idx=cs["heat"],
            N_uptake=upN, P_uptake=upP, K_uptake=upK, N_fixed=cs["fixN"], N_leach=cs["nleach"], erosion_t_ha=cs["eros"],
            soc_start_pct=cs["soc0"] / TOPSOIL_MASS_KG * 100, soc_end_pct=st["soc"] / TOPSOIL_MASS_KG * 100,
            ph_start=cs["ph0"], ph_end=st["ph"], dis_S=cs["S"], dis_loss=dis_loss, pest_dmg=cs["pest_dmg"],
            survival=cs["surv"], ky_pen=ky_pen, n_fert=m.n_kg, p_fert=m.p_kg, k_fert=m.k_kg, lime_t=m.lime_t,
            wue_kg_mm=ydm / max(cs["Tr"] + cs["E"], 1.0),
        )
