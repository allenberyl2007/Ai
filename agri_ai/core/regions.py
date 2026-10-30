"""Régions RDC et archétypes de sols.
ATTENTION : les normales climatiques ci-dessous sont des ordres de grandeur issus de la climatologie
générale (à CALIBRER sur NASA POWER / CHIRPS dès que les données réelles sont disponibles —
c'est précisément le rôle du mode 'nasa'). Elles ne sont pas des mesures."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Region:
    name: str
    lat: float
    lon: float
    elev: float                      # m
    rain_mm: Tuple[float, ...]       # normales mensuelles (12)
    tmean: Tuple[float, ...]         # °C (12)
    dtr: Tuple[float, ...]           # amplitude diurne °C (12)
    rain_cv: float = 0.15            # CV interannuel de la pluie annuelle
    soils: Tuple[str, ...] = ()      # sols typiques


REGIONS = {
    "kinshasa": Region("kinshasa", -4.33, 15.31, 310,
        (135, 145, 190, 200, 120, 5, 3, 6, 30, 130, 220, 150),
        (26.5, 26.8, 27.0, 27.0, 26.8, 24.8, 23.8, 24.5, 26.0, 26.5, 26.3, 26.0),
        (7, 7, 7, 7, 7, 8, 8, 9, 9, 8, 7, 7), 0.15, ("arenosol_sand", "ferralsol_scl")),
    "kisangani": Region("kisangani", 0.52, 25.20, 400,
        (90, 110, 150, 170, 170, 120, 110, 160, 190, 220, 180, 110),
        (25.5, 25.8, 25.8, 25.6, 25.4, 24.9, 24.4, 24.5, 24.9, 25.0, 25.1, 25.2),
        (9, 9, 8, 8, 8, 7, 7, 7, 7, 7, 8, 8), 0.10, ("acrisol_cl", "ferralsol_scl")),
    "bukavu": Region("bukavu", -2.50, 28.85, 1500,
        (140, 150, 160, 150, 70, 10, 5, 25, 90, 150, 170, 150),
        (19.8, 19.9, 19.8, 19.6, 19.3, 18.8, 18.5, 19.2, 19.8, 19.9, 19.6, 19.6),
        (9, 9, 8, 8, 9, 10, 10, 10, 9, 8, 8, 8), 0.15, ("nitisol_clay", "andosol_loam")),
    "lubumbashi": Region("lubumbashi", -11.66, 27.48, 1230,
        (220, 190, 160, 50, 8, 1, 0.5, 1, 8, 50, 150, 230),
        (21.5, 21.5, 21.5, 21.0, 18.5, 16.0, 16.0, 18.0, 21.0, 23.0, 22.5, 21.5),
        (8, 8, 9, 10, 12, 13, 13, 13, 13, 12, 9, 8), 0.18, ("ferralsol_scl", "arenosol_sand")),
    "kananga": Region("kananga", -5.90, 22.40, 650,
        (150, 150, 190, 190, 120, 5, 3, 25, 100, 190, 220, 170),
        (24.8, 25.0, 25.0, 24.9, 24.5, 23.2, 22.8, 23.8, 24.6, 24.8, 24.5, 24.5),
        (8, 8, 8, 8, 8, 9, 10, 10, 9, 8, 8, 8), 0.15, ("ferralsol_scl", "arenosol_sand")),
    "mbandaka": Region("mbandaka", 0.05, 18.26, 350,
        (100, 100, 150, 170, 160, 100, 100, 140, 180, 210, 190, 130),
        (25.8, 26.0, 26.0, 25.9, 25.7, 25.2, 24.8, 24.9, 25.2, 25.4, 25.5, 25.6),
        (8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 8), 0.10, ("acrisol_cl", "fluvisol_loam")),
}


@dataclass(frozen=True)
class SoilType:
    name: str
    wp: float; fc: float; sat: float     # m3/m3
    drain: float                         # fraction du surplus (> FC) drainé par jour
    cn: float                            # SCS curve number (condition II)
    k_erod: float                        # erodibilité (t ha h / ha MJ mm)
    soc: float                           # % SOC horizon 0-30 cm
    ph: float
    p_bray: float                        # ppm
    k_exch: float                        # cmol/kg
    cec: float                           # cmol/kg
    p_fix: float                         # fraction de P engrais immobilisée (fixation)
    c_n: float = 12.0


SOILS = {
    "arenosol_sand":  SoilType("arenosol_sand", 0.05, 0.12, 0.40, 0.90, 68, 0.020, 0.7, 5.0, 6.0, 0.08, 3.0, 0.20),
    "ferralsol_scl":  SoilType("ferralsol_scl", 0.17, 0.28, 0.45, 0.50, 78, 0.012, 1.4, 4.8, 4.0, 0.15, 5.0, 0.60),
    "nitisol_clay":   SoilType("nitisol_clay", 0.22, 0.36, 0.50, 0.35, 80, 0.015, 3.0, 5.6, 12.0, 0.40, 18.0, 0.55),
    "andosol_loam":   SoilType("andosol_loam", 0.20, 0.40, 0.55, 0.40, 72, 0.010, 6.0, 5.8, 8.0, 0.50, 25.0, 0.80),
    "fluvisol_loam":  SoilType("fluvisol_loam", 0.12, 0.30, 0.46, 0.30, 82, 0.030, 2.2, 6.0, 15.0, 0.35, 12.0, 0.30),
    "acrisol_cl":     SoilType("acrisol_cl", 0.16, 0.30, 0.46, 0.45, 80, 0.025, 1.8, 4.5, 3.0, 0.12, 6.0, 0.60),
}

# Profondeurs de couches (m) : topsoil / subsoil
Z1, Z2 = 0.30, 0.70
BULK_DENSITY = 1.3                      # g/cm3
TOPSOIL_MASS_KG = BULK_DENSITY * 1000 * Z1 * 10000   # kg de sol / ha (0-30 cm) = 3.9e6


# Fenêtres de semis (jour de l'année de début, durée de la fenêtre en jours) par saison culturale.
# Déduites de la climatologie ci-dessus (début des pluies principales); à recalibrer sur CHIRPS/POWER.
SEASONS = {
    "kinshasa":   ((274, 45), (46, 30)),    # A: début oct ; B: mi-fév (courte)
    "kisangani":  ((60, 40), (244, 40)),    # A: mars ; B: sept
    "bukavu":     ((258, 40), (46, 40)),    # A: mi-sept ; B: mi-fév
    "lubumbashi": ((319, 40),),             # une saison : mi-nov
    "kananga":    ((268, 40), (51, 30)),    # A: fin sept ; B: fin fév
    "mbandaka":   ((60, 40), (244, 40)),
}
