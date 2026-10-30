"""Importe dans simulation.db des fichiers téléchargés À LA MAIN (sans API) : CSV NASA POWER (mensuel ou journalier) ou export FAOSTAT.
Usage : python scripts/import_csv.py power  <nom_de_site>  fichier.csv
        python scripts/import_csv.py faostat fichier.csv"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.db import connect
from adapters import power, faostat

con = connect()
if sys.argv[1] == "power": print(power.load_csv(con, sys.argv[2], sys.argv[3]))
elif sys.argv[1] == "faostat": print(faostat.load_csv(con, sys.argv[2]), "lignes lues")
else: sys.exit(__doc__)
