# README of SpecSimulator

## Pour faire un set de simu

```python
python SpecSimulator/main_simu.py f=<nom_dossier> nsimu=<nombre_simu> tel=<telescope>
```

* ***f*** est le nom du dossier
* ***nsimu*** est le nombre de simu
* ***tel*** est le nom ud telescope : *ctio*, *stardice* (instable), *auxtel*, *auxtelqn* (auxtel quad notch)

Ceci va créer un dossier `results/output_simu/<nom_dossier>`, avec à l'interieur:
* *spectrum*, dossier avec des npy contenant le vecteur d'intensité du spectre (longueur d'onde entre 300 et 1100 par pas de 1)
* *image*, dossier avec des npy contenant les simulation sur le CCD, respectivement à chaque spectre de *spectrum*
* *hparams.json*, json avec les parametres de la simulation
* *vparams.npz*, liste des parametres variables et unique de chaque simulation (airmass, target, etc.)
* *imageOrigin*, si le tel est auxtel ou auxtelqn, les images dans *image* sont rebin, donc elle sont en entier dans *imageOrigin*