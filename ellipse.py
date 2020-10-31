"""
Régression elliptique se basant sur la méthode des moindres carrés totaux
voir : https://fr.wikipedia.org/wiki/R%C3%A9gression_elliptique#M%C3%A9thode_des_moindres_carr%C3%A9s_totaux
"""

from cercle import regression_circulaire

import numpy as np
import scipy.optimize as optimize
from skimage.color import rgb2gray
from skimage import draw
import matplotlib.pyplot as plt


def phis(x, y, cx, cy, theta):
    """
    Initialise les angles de l'équation paramétrique de chaque point
    pour calculer les vecteurs erreur
    Entrées :
        - x, y : points pour la régression (taille N)
        - cx, cy : centre de l'ellipse
        - theta : inclinaison de l'ellipse
    Sortie : angles en radians (taille N)
    """
    t = np.arctan2(y - cy, x - cx) - theta
    return t


def equParametrique_ell(cx, cy, a, b, theta, t):
    '''
    Calcule les coordonnées d'un point sur une ellipse paramétrée
    Entrées :
        - cx, cy : centre de l'ellipse
        - a, b : longueurs des demi-grands axes
        - theta : inclinaison
        - t : angle du point (peut être un vecteur d'angless)
    Sortie : coordonnée du point
    '''
    centre = np.array([[cx], [cy]])
    mat_rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    coords_ellipse = centre + mat_rot @ np.vstack((a * np.cos(t), b * np.sin(t)))
    return coords_ellipse


def residus(z, x, y):
    """
    Calcule les résidus
    Entrées : 
        - z : paramètres de l'ellipse (centre, demi axes, inclinaisons, angles)
        - x, y : coordonnées des points
    Sortie : résidus au carré
    """
    cx, cy, a, b, theta, phis = z[0], z[1], z[2], z[3], z[4], z[5:]
    coords_ellipse = equParametrique_ell(cx, cy, a, b, theta, phis)
    vecteurs_erreur = coords_ellipse - np.vstack((x, y))
    residus = np.linalg.norm(vecteurs_erreur, axis=0)

    return np.square(residus)


def masque_ell(xe, ye, xd, yd, R):
    """
    Calcule à partir du centre (xe,ye) de l'ellipse,
    du centre (xd,yd) et du rayon R du disque lunaire
    l'inclinaison theta et l'aplatissement de l'ellipse
    """
    x_prime = xe - xd
    y_prime = yd - ye
    cos_beta = np.sqrt(R ** 2 - np.square(x_prime) - np.square(y_prime)) / R
    theta = np.arctan2(x_prime, y_prime)
    return theta, cos_beta


def residus2(z, x, y, xd, yd, R):
    """
    Calcule les résidus avec moins de paramètres de régression
    Entrées : 
        - z : paramètres de l'ellipse (centre, un seul demi axe, angles)
        - x, y : coordonnées des points
    Sortie : résidus au carré
    """
    cx, cy, a, phis = z[0], z[1], z[2], z[3:]

    theta, cos_beta = masque_ell(cx, cy, xd, yd, R)
    b = a * cos_beta

    coords_ellipse = equParametrique_ell(cx, cy, a, b, theta, phis)
    vecteurs_erreur = coords_ellipse - np.vstack((x, y))
    residus = np.linalg.norm(vecteurs_erreur, axis=0)

    return np.square(residus)


def regression_elliptique(x, y, cx=0, cy=0, a=1, b=1, theta=0, phis=None):
    """
    Régression elliptique avec comme variables le centre de l'ellipse, ses demi-axes et son inclinaison
    Entrées :
        - x, y : points qui vont servir à la régression
    Sorties :
        - res : résultat de la régression (centre de l'ellipse et demi-grand axe "horizontal")
    """
    # Initialisation des paramètres
    N = len(x)
    if phis is None: phis = np.zeros((N))
    z0 = np.append([cx, cy, a, b, theta], phis)

    # méthode des moindres carrés
    bounds = (
        [-np.inf] * 4 + [-np.pi] * (N + 1),
        [np.inf] * 4 + [np.pi] * (N + 1),
    )  # je sais pas si cest très utile de restreindre les paramètres...
    res = optimize.least_squares(residus, z0, args=(x, y), bounds=bounds, max_nfev=50)
    return res


def regression_elliptique2(x, y):
    """
    Régression elliptique avec comme variables le centre de l'ellipse, ses demi-axes et son inclinaison
    On initialise les paramètres de la régression elliptique
    en faisant d'abord une régression circulaire
    Entrées :
        - x, y : points qui vont servir à la régression
    Sorties :
        - res : résultat de la régression (centre de l'ellipse et demi-grand axe "horizontal")
    """

    # Initialisation des paramètres pour la régression circulaire
    c0 = np.array([(x.min() + x.max()) / 2, (y.min() + y.max()) / 2])
    res, r = regression_circulaire(c0, x, y)

    # Initialisation des paramètres pour la régression elliptique
    cx0, cy0 = res.x
    a0 = r
    b0 = r
    theta0 = 0
    phis0 = phis(
        x, y, cx0, cy0, 0
    )  # Voir wiki : "Pour initialiser φi, on peut utiliser l'angle par rapport à l'axe x du segment reliant le centre initial de l'ellipse au point expérimental i. "

    res_ellipse = regression_elliptique(x, y, cx0, cy0, a0, b0, theta0, phis0)
    return res_ellipse


def regression_elliptique3(x, y, xd, yd, R):
    """
    Dans cette regression elliptique, la fonction des résidus n'est plus la même
    residus2 respecte la forme des cratères (voir pdf JLH "masque de visée")
    theta et b ne sont plus des variables :
        - theta (inclinaison de l'ellipse) est choisi tel que l'ellipse soit dirigée vers le centre du disque
        - b dépend de la distance de l'ellipse au centre du disque : b = a*cos(beta) (voir calcul de cos(beta) dans le pdf de JLH)
    Entrées :
        - x, y : points qui vont servir à la régression
        - xd, yd : centre du disque de la Lune
        - R : rayon du disque Lunaire
    Sorties :
        - res : résultat de la régression (centre de l'ellipse et demi-grand axe "horizontal")
        - b : longueur demi-grand axe "vertical"
        - theta : inclinaison de l'ellipse
    """

    # Initialisation des variables pour la régression elliptique
    # On commence par une régression circulaire
    c0x, c0y = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    c0 = np.array([c0x, c0y])
    res, r = regression_circulaire(c0, x, y)

    cx0, cy0 = res.x
    theta, _ = masque_ell(cx0, cy0, xd, yd, R)
    a0 = r
    phis0 = phis(x, y, c0x, c0y, theta)

    z0 = np.append([c0x, c0y, a0], phis0)
    N = len(x)
    # On définit le domaine des varaibles
    bounds = ([0, 0, 0] + [-np.inf] * N, [np.inf, np.inf, R] + [np.inf] * N)

    # Méthode des moindres carrés
    try:
        res = optimize.least_squares(
            residus2,
            z0,
            args=(x, y, xd, yd, R),
            bounds=bounds,
            verbose=0,
            xtol=1e-06,
            max_nfev=100,
        )
        # Mise en forme des résultats
        cx, cy, a, _ = res.x[0], res.x[1], res.x[2], res.x[3:]
        theta, cos_beta = masque_ell(cx, cy, xd, yd, R)
        b = a * cos_beta
    except ValueError:
        print("Problème dans la détection de l'ellipse")
        return None, None, None

    return res, b, theta