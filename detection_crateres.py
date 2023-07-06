"""
Lance une interface graphique pour le repérage des cratères par l'utilisateur
"""

from ellipse import regression_elliptique3, regression_elliptique2
from detection_disque import detection_disque

import numpy as np
from skimage import io
from skimage.color import rgb2hsv, gray2rgb
from skimage import feature, measure, draw
from skimage import exposure
import matplotlib.pyplot as plt

from random import shuffle
from time import time
import re


def detection_crateres(moon_original):
    """
    Lance l'interface graphique
    Entrée : image de Lune (MxNx1)
    Sortie :
        - Liste des codes UAI des cratères identifiés
        - Liste des informations utiles sur les cratères (centre_y, centre_x, b, a, inclinaison) (pxl)
        - Centre du disque lunaire en y (pxl)
        - Centre du disque lunaire en x (pxl)
        - Rayon du disque lunaire (pxl)
    """
    # moon de taille [MxNx1]
    # renvoie une liste de cratères avec leur centre en pixel + centre-rayon du disque

    moon = moon_original.copy()
    moon_light = rgb2hsv(moon)[:, :, 2]
    moon_light = exposure.rescale_intensity(moon_light, in_range=(0.4, 1))

    # On trouve d'abord le disque lunaire
    print("--- Détection du disque lunaire ---")
    t0 = time()
    xc, yc, R = detection_disque(moon_light)
    resolution = 1730 / R  # en km/px
    print("Centre du disque, rayon (pxl) : ", xc, yc, R)
    # print("Détection disque : ", time()-t0)
    t0 = time()

    part_surface = 0.9

    mask_r, mask_c = draw.disk(
        (np.around(yc).astype(int), np.around(xc).astype(int)), R*part_surface, shape=moon.shape
    )  # le masque est un peu plus petit que le disque (0.8*R)
    mask = np.zeros_like(moon_light, dtype="bool")
    mask[mask_r, mask_c] = True
    # print("Création d'un masque : ", time()-t0)

    # On estime la proportion des contours des cratères par rapport à la surface totale de la Lune
    ratio = (
        72 * part_surface * part_surface
    )  # estimation de : longueur périmètres cratères (km) / diamètre Lune (km)
    perimetre_tot = ratio * (R * 2)  # estimation perimetre tot de cratère
    surface = np.pi * R * R
    frac = perimetre_tot / surface
    frac_cible = frac * 0.3
    # print("Fraction de pixels à considérer : ", frac_cible)

    # Détecte des contours (on pourra modifier les parametres)
    print("--- Détection des contours ---")
    t0 = time()
    edges = feature.canny(
        moon_light,
        sigma=R / 1000,
        low_threshold=1 - frac_cible,
        high_threshold=1 - frac_cible / 4,
        use_quantiles=True,
        mask=mask,
    )
    print("Application du filtre de canny (s) : ", time() - t0)
    plt.show()

    # Décompose les contours par composantes connexes
    # labels est une liste de contours qui sont séparés, num est le nombre de contours différents qu'on a détecté
    t0 = time()
    labels, num = measure.label(edges, return_num=True, connectivity=2)
    # print("Décomposition en composantes connexes : ", time()-t0)

    S0 = 0
    S1, i = 0, 0
    S2, j = 0, 0

    keys = []
    crat_infos = []
    liste_crateres = list(range(1, num))

    # Début de la détection semi-automatique des cratères
    print(
        "--- Début de la détection semi-automatique des cratères ---\nMode 1 : les cratères sont affichés un par un. (long)\nMode 2 : les cratères sont affichés en même temps\n Mode 2 conseillé"
    )
    mode = None
    while mode != "1" and mode != "2":
        mode = input("Mode ? (1/2) ")
        if mode == "1":
            shuffle(liste_crateres)
        elif mode == "2":
            pass
        else:
            print("rep must be 1 or 2")
    moon3 = moon_original.copy()
    num_crat = 1
    for l in liste_crateres:
        # print('----------')
        print(l, "/", num)
        t0 = time()
        y, x = np.nonzero(labels == l)
        S0 += time() - t0
        t0 = time()
        res, b, theta = regression_elliptique3(x, y, xc, yc, R)
        if res is not None:
            c_x, c_y, a = res.x[:3]
            # res = regression_elliptique2(x,y)
            # c_x, c_y, a, b, theta = [res.x[i] for i in range(5)]
            ecart_type = np.sqrt(res.cost / len(x))

            S1 += time() - t0
            i += 1
            if (
                ecart_type * resolution < 1 and a * resolution < 20
            ):  # l'écart type est de moins de 100 km
                print("Cratère détecté, diamètre du cratère : ", a * resolution)
                t0 = time()
                if mode == "1":
                    ax = plt.subplot()
                    moon3 = moon_original.copy()
                    ax.text(int(c_x), int(c_y), "Ici")
                else:
                    plt.text(int(c_x), int(c_y), num_crat)
                    num_crat += 1
                ell_y, ell_x = draw.ellipse_perimeter(
                    int(c_y),
                    int(c_x),
                    int(b),
                    int(a),
                    orientation=theta,
                    shape=moon.shape,
                )
                moon3[ell_y, ell_x] = [0, 255, 0]
                if 0 <= int(c_y) < moon3.shape[0] and 0 <= int(c_x) < moon3.shape[1]:
                    moon3[int(c_y), int(c_x)] = [0, 255, 0]

                if mode == "1":
                    io.imshow(moon3)
                    plt.show()

                S2 += time() - t0
                j += 1

                if mode == "1":
                    rep = input(
                        "Voulez-vous garder ce cratère (o/n ou q pour quitter) ? "
                    )
                    if rep == "o":
                        pattern = re.compile("[A-Z]{2,2}[0-9]{4,4}[A-Z][0-9]{5,5}[A-Z]")
                        res = None
                        while res is None:
                            code = input("code L.U.N du cratère ? ").upper().strip()
                            res = pattern.search(code)
                        code = res[0]
                        print(code)
                        keys.append(code)
                        crat_infos.append((c_y, c_x, b, a, theta))
                        print("Cratère ajouté !")
                    elif rep == "q":
                        break
                else:
                    crat_infos.append((c_y, c_x, b, a, theta))

    if mode == "2":
        rep = None
        crat_infos2 = []
        while rep != "Q":
            io.imshow(moon3)
            num_crat = 1
            for crat in crat_infos:
                c_y, c_x, _, _, _ = crat
                plt.text(int(c_x), int(c_y), num_crat)
                num_crat += 1
            plt.show()
            rep = (
                input(
                    "Entrez le numéro du cratère suivi de son code LUN (numéro,code) ou q pour terminer\n"
                )
                .upper()
                .split(",")
            )
            pattern_code = re.compile("[A-Z]{2,2}[0-9]{4,4}[A-Z][0-9]{5,5}[A-Z]")
            pattern_num = re.compile("[0-9]*")
            if len(rep) > 1:
                num = pattern_num.search(rep[0])
                code = pattern_code.search(rep[1])
                if num is None or code is None:
                    print("Pattern not found")
                else:
                    num = num[0]
                    code = code[0]
                    num = int(num) - 1
                    if num >= 0 and num < len(crat_infos):
                        keys.append(code)
                        crat_infos2.append(crat_infos[num])
                        print(
                            "Cratère ajouté ! ", len(keys), " cratère(s) enregistré(s)."
                        )
                    else:
                        print("Numéro incorrect")
            elif rep[0] != "Q":
                print("Pas compris, réessayez")
            elif rep[0] == "Q":
                rep = rep[0]
        crat_infos = list(crat_infos2)

    print("Terminé !\n")

    crat_infos = np.array(crat_infos)
    return keys, crat_infos, yc, xc, R