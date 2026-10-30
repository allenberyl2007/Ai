"""Paramètres culturaux (valeurs d'ordre de grandeur issues de FAO-56/FAO-33, AquaCrop, littérature
agronomique tropicale). Tous ces paramètres sont des HYPOTHÈSES A CALIBRER sur données réelles
(essais IITA/INERA, FAOSTAT) — voir audit."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Crop:
    name: str
    tbase: float; topt: float; tmax: float       # températures cardinales (degC)
    gdd_mat: float                               # degrés-jours jusqu'à maturité
    rue: float                                   # g MS / MJ Rs intercepté
    hi: float                                    # indice de récolte (organe récolté / biomasse totale)
    cc_max: float                                # couverture maximale
    kcb_ini: float; kcb_mid: float; kcb_end: float
    root_max: float                              # m
    p_dep: float                                 # fraction d'épuisement tolérée (FAO-56)
    ky_flow: float                               # sensibilité du HI au stress hydrique en floraison
    conc: Tuple[float, float, float]             # N, P, K min dans la biomasse (kg/kg MS)
    hi_nut: Tuple[float, float, float]           # part de N,P,K de la plante dans l'organe récolté
    fix_frac: float                              # fraction de N issue de la fixation biologique (légumineuses)
    ph_crit: float; ph_sens: float               # pH critique (toxicité Al) et sensibilité
    heat_thr: float                              # Tmax seuil de stress thermique floraison
    wl_sens: float; wl_tol_days: float           # sensibilité / tolérance à l'engorgement
    moisture: float                              # humidité de référence du rendement commercial
    family: str                                  # famille pour rotation (inoculum)
    dis_susc: float; dis_maxloss: float; dis_T: Tuple[float, float]; dis_rh: float
    pest_prob: float                             # probabilité de base d'un épisode ravageur/saison
    res_net_n: float                             # fraction nette de N résiduel minéralisé
    unit: str


CROPS = {
    "maize": Crop("maize", 8, 30, 42, 1900, 1.75, 0.45, 0.90, 0.15, 1.15, 0.50, 1.0, 0.55, 1.5,
                  (0.010, 0.0012, 0.009), (0.70, 0.75, 0.25), 0.0, 5.3, 0.9, 35.0, 0.6, 3.0, 0.15,
                  "cereal", 0.8, 0.35, (18, 28), 85, 0.35, 0.0, "t/ha grain 15% hum."),
    "cassava": Crop("cassava", 12, 28, 38, 3400, 1.05, 0.55, 0.85, 0.20, 0.90, 0.60, 1.2, 0.60, 0.5,
                    (0.008, 0.0008, 0.010), (0.45, 0.50, 0.55), 0.0, 4.4, 0.3, 38.0, 0.9, 3.0, 0.65,
                    "cassava", 1.0, 0.45, (22, 32), 70, 0.25, 0.3, "t/ha racines fraîches"),
    "bean": Crop("bean", 8, 24, 35, 1050, 1.35, 0.40, 0.85, 0.15, 1.05, 0.40, 0.7, 0.45, 1.15,
                 (0.025, 0.0020, 0.015), (0.70, 0.75, 0.40), 0.50, 5.2, 1.0, 32.0, 0.9, 2.0, 0.14,
                 "legume_a", 0.9, 0.40, (16, 26), 88, 0.30, 0.6, "t/ha grain 14% hum."),
    "groundnut": Crop("groundnut", 10, 28, 40, 1500, 1.30, 0.38, 0.85, 0.15, 1.05, 0.55, 0.9, 0.50, 0.7,
                      (0.030, 0.0020, 0.012), (0.65, 0.75, 0.30), 0.50, 5.0, 0.6, 36.0, 0.8, 2.0, 0.10,
                      "legume_b", 0.8, 0.40, (20, 30), 80, 0.20, 0.6, "t/ha gousses sèches"),
    "rice_upland": Crop("rice_upland", 10, 30, 40, 2000, 1.50, 0.40, 0.90, 0.15, 1.15, 0.50, 0.9, 0.20, 1.1,
                        (0.010, 0.0015, 0.011), (0.65, 0.75, 0.20), 0.0, 4.4, 0.3, 35.0, 0.2, 15.0, 0.14,
                        "rice", 0.9, 0.40, (22, 30), 90, 0.25, 0.0, "t/ha paddy 14% hum."),
}
