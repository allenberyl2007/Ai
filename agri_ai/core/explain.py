"""Explications en français simple. Les raisons viennent de règles lisibles appliquées aux entrées/sorties
des modèles (pas d'une boîte noire)."""
from .mlkit import prob_exceed
from .policy import FR


def level(p, lo=0.2, hi=0.5):
    return "faible" if p < lo else ("moyen" if p < hi else "élevé")


def explain(crop, soil, mgmt, clim, soilout, cropout, yld, rain_norm):
    """clim/soilout/cropout/yld : dicts {cible: (lo, mid, hi)}. Retourne une liste de phrases."""
    out = []
    low_n = soil["soc_pct"] < 1.5 and mgmt["n_kg"] < 30
    if low_n:
        out.append("Le sol manque d'azote." if crop not in ("bean", "groundnut") else "Le sol manque d'azote, cette plante en a besoin de peu.")
    if crop in ("bean", "groundnut"):
        out.append(f"Le {FR[crop].lower()} améliore naturellement le sol : il prend l'azote de l'air." if crop == "bean" else
                   "L'arachide améliore naturellement le sol : elle prend l'azote de l'air.")
    if soil["ph"] < 5.0 and crop in ("maize", "bean"):
        out.append("Le sol est assez acide, ce qui gêne un peu cette culture (de la chaux aiderait).")
    p_dry = prob_exceed(14, *clim["dry_spell_days"]); p_low = prob_exceed(0.75 * rain_norm, *clim["season_rain_mm"])
    p_dry_season = 1 - p_low
    sec = max(p_dry, p_dry_season)
    out.append(f"Le risque de sécheresse est {level(sec)}.")
    p_flood = prob_exceed(100, *clim["max3_rain_mm"])
    if p_flood > 0.2: out.append(f"Le risque de fortes pluies et d'inondation est {level(p_flood, 0.3, 0.6)}.")
    e = soilout["erosion_t_ha"][1]
    out.append("Cette culture protège bien le sol contre l'érosion." if e < 5 else
               ("Le sol risque de partir avec la pluie (érosion moyenne) : planter en travers de la pente aide." if e < 20 else
                "Attention : forte perte de terre par la pluie sur ce terrain en pente."))
    need = cropout["water_use_mm"][1]
    out.append(f"Besoin en eau : {level(need / max(rain_norm, 1), 0.45, 0.75)}.")
    return out
