#!/bin/bash
# Entraîne les 5 modèles dans l'ordre (la Recommendation AI a besoin des erreurs mesurées des 4 autres).
cd "$(dirname "$0")/.."
for m in climate soil crop yield recommendation; do echo "=== $m ==="; python3 models/$m/train.py || exit 1; done
