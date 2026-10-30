"""Règle de décision explicite et modifiable : « valeur » d'une culture pour un petit exploitant.
Valeur = millions de kcal par hectare et par 100 jours d'occupation du terrain, réduite par les dégâts au sol
et le stress hydrique. Aucun prix de marché n'est utilisé (non disponible) : à remplacer par des prix FAOSTAT/locaux."""
KCAL_PER_KG = {"maize": 3500, "cassava": 1600, "bean": 3300, "groundnut": 4000, "rice_upland": 2500}  # ordres de grandeur
CROPS = list(KCAL_PER_KG)
FR = {"maize": "Maïs", "cassava": "Manioc", "bean": "Haricot", "groundnut": "Arachide", "rice_upland": "Riz pluvial"}


def score(crop, yield_t, days, erosion_t, d_soc, stress):
    kcal_m = yield_t * KCAL_PER_KG[crop] / 1000.0                       # Mkcal/ha sur le cycle
    per100 = kcal_m / max(days, 30.0) * 100.0
    pen = min(0.3, max(0.0, erosion_t) / 50.0) + (0.1 if d_soc < -0.05 else 0.0) + min(0.2, max(0.0, stress - 0.1))
    return per100 * (1.0 - pen)
