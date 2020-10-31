"""
Détection des contours et régression circulaire
voir : https://fr.wikipedia.org/wiki/R%C3%A9gression_circulaire#M%C3%A9thode_des_moindres_carr%C3%A9s_totaux
"""

from cercle import regression_circulaire

import numpy as np
from skimage import feature, draw, io
from skimage.color import rgb2hsv
from skimage import draw, measure
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from skimage.exposure import histogram


def detection_disque(moon):
    """
    Détection du contour de la Lune
    Entrée : image de Lune (MxNx1)
    Sorties :
        - xc, yc : cordonnées du centre du disque en pixel
        - r : rayon du disque en pixels
    """

    # On détecte le contour de la Lune avec un filtre de Canny
    # edges est une image binaire, 1 signifie qu'il y a un contour, 0 qu'il n'y a rien
    D_approx = min(moon.shape)
    target = (
        np.pi * D_approx / (D_approx ** 2)
    )  # estimation de la part du périmètre dans l'image
    # On applique un filtre de Canny en sélectionnant le moins de pixels possibles en utilisant les quantiles
    # On a : len(edges) < 2*target*nb_pixels_image

    edges = feature.canny(
        moon,
        sigma=D_approx / 100,
        low_threshold=1 - target * 4,
        high_threshold=1 - target,
        use_quantiles=True,
    )
    edges_indices = np.array(np.nonzero(edges))

    # Le problème est qu'on peut détecter des contours qui ne nous intéressent pas (contours de cratères, contour de la ligne jour-nuit)

    # On garde seulement les contours assez longs (n > moon.shape[0]/10)
    # Ca permet d'éliminer les contours isolés (type cratère) mais on a parfois encore la ligne jour-nuit qui est en partie détectée comme un contour...
    labels, num = measure.label(edges, return_num=True, connectivity=2)
    if num == 1:
        edges_selected = edges
    else:
        edges_selected = np.zeros(moon.shape)
        for l in range(1, num):
            contour_l = labels == l
            indices_contour = np.nonzero(contour_l)
            n = len(indices_contour[0])
            if n > D_approx * np.pi / 8:
                edges_selected += labels == l

    edges_indices = np.array(np.nonzero(edges_selected))

    # on récupère les coordonnées des contours sélectionnés, ils sont le contour du disque
    edges_indices = np.array(np.nonzero(edges_selected))

    # On fait une regression circulaire avec le centre c0 (le point d'initialisation de l'algo) au milieu de l'image
    c0 = np.array([moon.shape[1] // 2, moon.shape[0] // 2])
    y = edges_indices[0, :]
    x = edges_indices[1, :]
    res, r = regression_circulaire(c0, x, y)
    xc, yc = res.x[0], res.x[1]

    return xc, yc, r


def phase(moon):
    """
    Calcule la phase de la Lune avec un treshold
    Entrée : photo de la Lune (MxNx1)
    Sortie : phase en rad
    """
    moon_light = rgb2hsv(moon)[:, :, 2]
    xc, yc, R = detection_disque(moon_light)
    print("rayon=", R)
    eclaire = moon_light > 0.2
    moon_light[eclaire] = 1
    io.imshow(moon_light)
    plt.show()
    phase = len(moon_light[eclaire]) / (np.pi * R * R)
    # print(prop_eclaire)
    # phase = (2/np.pi)*np.arccos(1-prop_eclaire)
    return phase
