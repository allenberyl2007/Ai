# Exemples de sortie (météo simulée, mode sim)

```
python main.py --region kananga --soil ferralsol_scl --slope 8 --prev bean --demo

Culture recommandée :

Maïs

Pourquoi ?

Le sol manque d'azote.
Le sol est assez acide, ce qui gêne un peu cette culture (de la chaux aiderait).
Le risque de sécheresse est faible.
Le risque de fortes pluies et d'inondation est moyen.
Cette culture protège bien le sol contre l'érosion.
Besoin en eau : moyen.
Récolte attendue : environ 1.6 tonnes par hectare (entre 1.1 et 1.9).

Confiance :

68 %

Alternative :

Haricot

Note : Le système n'a été entraîné que sur des simulations : confiance limitée.
```

```
python main.py --region bukavu --soil nitisol_clay --slope 25 --prev none --n 40 --p 15 --demo

Culture recommandée :

Haricot

Pourquoi ?

Le haricot améliore naturellement le sol : il prend l'azote de l'air.
Le risque de sécheresse est moyen.
Le risque de fortes pluies et d'inondation est faible.
Attention : forte perte de terre par la pluie sur ce terrain en pente.
Besoin en eau : moyen.
Récolte attendue : environ 2.7 tonnes par hectare (entre 1.8 et 3.0).

Confiance :

44 %

Alternative :

Maïs

Réponse peu sûre : demandez aussi conseil à un technicien agricole.

Note : Le système n'a été entraîné que sur des simulations : confiance limitée.
```

```
python main.py --region lubumbashi --soil arenosol_sand --slope 3 --prev maize --demo

Culture recommandée :

Haricot

Pourquoi ?

Le sol manque d'azote, cette plante en a besoin de peu.
Le haricot améliore naturellement le sol : il prend l'azote de l'air.
Le risque de sécheresse est faible.
Le risque de fortes pluies et d'inondation est moyen.
Cette culture protège bien le sol contre l'érosion.
Besoin en eau : faible.
Attention : la récolte attendue est très faible, même pour la meilleure culture. Améliorer le sol (engrais, chaux, compost) aiderait beaucoup.
Récolte attendue : environ 0.2 tonnes par hectare (entre 0.1 et 0.3).

Confiance :

40 %

Alternative :

Maïs

Réponse peu sûre : demandez aussi conseil à un technicien agricole.

Note : Le système n'a été entraîné que sur des simulations : confiance limitée.
Note : Haricot et Maïs sont très proches : les deux sont de bons choix.
```
