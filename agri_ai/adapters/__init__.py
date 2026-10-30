"""Un adaptateur par source : download() -> brut ; parse() -> DataFrame canonique ; load(con) -> table SQLite.
IMPORTANT : aucun adaptateur n'a pu être testé contre son service réel (réseau désactivé lors de la construction).
Chaque parse() est testé sur des charges utiles fictives qui suivent le format documenté (tests/test_adapters.py)."""
import json, urllib.request


def http_json(url, data=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=headers or {})
    if data is not None: req.add_header("Content-Type", "application/json")
    return json.load(urllib.request.urlopen(req, timeout=timeout))
