import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.mlkit import Predictor

_P = None


def predict(df):
    """df : DataFrame avec (au moins) les colonnes listées dans meta.json['features'].
    Retourne <cible>_lo / _mid / _hi / _unc / _conf + ood_score / ood_flag / unseen_features."""
    global _P
    _P = _P or Predictor(os.path.dirname(os.path.abspath(__file__)))
    return _P.predict(df)
