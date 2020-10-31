"""
Lance la détection de cratères puis le calcul de l'état de libration
"""

from detection_crateres import detection_crateres
from libration import trouver_libration
from data import getDims, loadData
from draw_moon import draw_final

import argparse
from skimage import io, color
import matplotlib.pyplot as plt


def main(img_path, db_path):
    """
    Lance la détection de cratères puis le calcul de l'état de libration.
    Affiche le résultat sur une image avec matplotlib
    Entrée :
        - img_path : chemin vers l'image de Lune
        - db_path : chemin vers la base de données de cratères
    """
    cratereDB = loadData(db_path)
    moon_original = io.imread(img_path)
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lance la détection de cratères et le calcul de la libration d'une photo de Lune"
    )
    parser.add_argument(
        "--img_path",
        required=False,
        default="images/Lune_Audierne.jpg",
        help="chemin vers la photo",
    )
    parser.add_argument(
        "--db_path",
        required=False,
        default="crateres.csv",
        help="chemin vers la base de données des cratères",
    )
    args = parser.parse_args()

    main(args.img_path, args.db_path)