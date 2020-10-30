# coding: utf-8

from sys import argv
from skimage import io, color
import matplotlib.pyplot as plt
from detection_crateres import detection_crateres
from libration import trouver_libration
from data import getDims, loadData
from draw_moon import draw_final


def main():
    """ Fonction principale """
    cratereDB = loadData("../data/crateres.csv")

    if len(argv) > 1:
        photo_filename = argv[1]
    else:
        photo_filename = "../images/Lune Audierne Philippe 01-06-2020 23h45.JPG"

    moon_original = io.imread(photo_filename)
    moon = moon_original.copy()
    print("Tailles de l'image : ", moon.shape[:2])

    codes, crat_infos, yc, xc, R = detection_crateres(moon)
    if len(codes) > 0:
        y, x = crat_infos[:, 0], crat_infos[:, 1]
        names, lat, lon, _ = getDims(cratereDB, codes)
        print("Liste des cratères renseignés :\n", names)

        res = trouver_libration(
            x, y, lon, lat, xc=xc, yc=yc, R=R, theta=0, Dphi=0, Dlambda=0
        )

        # On affiche le résultat
        ax = plt.subplot()
        moon = draw_final(ax, moon, cratereDB, res, x, y, lon, lat)
        io.imshow(moon)
        plt.show()

    else:
        print("Vous n'avez renseigné aucun cratère !")
