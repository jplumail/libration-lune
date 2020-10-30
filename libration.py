"""
Trouve l'état de libration de la Lune avec une méthode des moindres carrés
"""

import numpy as np
import scipy.optimize as optimize


def pxl_vers_selenographique(x, y, xc, yc, R, theta, Dphi, Dlambda):
    """
    Permet de passer du référentiel de la photo au référentiel lunaire
    Entrées :
        - x, y : position d'un point sur l'image (horizontale, verticale)
        - xc, yc, R : centre du disque lunaire et rayon du disque lunaire
        - theta : inclinaison de l'axe de rotation de la Lune par rapport à la verticale
        - Dphi, Dlambda : angles de libration
    Sorties :
        - phi, lambdaa : position sélénographique (latitude, longitude) du point x,y
    Calculs réalisés par Jean Le Hir
    """

    # centrage
    x, y = x - xc, yc - y

    # orientation
    R_theta = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    coord_incline = R_theta @ np.vstack((x, y))
    X_prime, Y_prime = coord_incline[0, :], coord_incline[1, :]

    # coordonnées sphériques
    phi_sphe = np.arcsin(Y_prime / R)
    lambda_sphe = np.arcsin(X_prime / (R * np.cos(phi_sphe)))

    # coordonnées sélénographique
    phi = np.arcsin(
        np.sin(phi_sphe) * np.cos(Dphi)
        + np.cos(phi_sphe) * np.sin(Dphi) * np.cos(lambda_sphe)
    )
    lambdaa = Dlambda + np.sign(X_prime) * np.arccos(
        (np.cos(lambda_sphe) * np.cos(phi_sphe) - np.sin(phi) * np.sin(Dphi))
        / (np.cos(phi) * np.cos(Dphi))
    )
    return phi, lambdaa


def sellenographique_vers_pxl(phi, lambdaa, xc, yc, R, theta, Dphi, Dlambda):
    """
    Permet de passer du référentiel lunaire au référentiel de la photo
    Entrées :
        - phi, lambdaa : position d'un point sur la Lune (latitude, longitude)
        - xc, yc, R : centre du disque lunaire et rayon du disque lunaire
        - theta : inclinaison de l'axe de rotation de la Lune par rapport à la verticale
        - Dphi, Dlambda : angles de libration
    Sorties :
        - x_pxl, y_pxl : position sur l'image du point phi, lambda
    Calculs réalisés par Jean Le Hir
    """

    # passage en coordonnées sphériques
    lambda2 = lambdaa - Dlambda
    x_prime = R * np.cos(phi) * np.sin(lambda2)
    y_prime = R * (
        np.sin(phi) * np.cos(Dphi) - np.cos(lambda2) * np.cos(phi) * np.sin(Dphi)
    )

    # rotation d'angle -theta
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coord_pxl_centree = rot @ np.vstack((x_prime, y_prime))

    # décentrage
    x_pxl, y_pxl = coord_pxl_centree[0, :] + xc, -coord_pxl_centree[1, :] + yc

    return x_pxl, y_pxl


def residus(z, x, y, lambdaa_cible, phi_cible):
    """
    Calcule les résidus pour la méthode des moindres carrés
    Entrées :
        - z : contient tous les paramètres à optimiser
        - x, y : position des cratères sur l'image (pixels)
        - lambdaa_cible, phi_cible : coordonnées sélénographiques des cratères
    Sorties :
        - residus_carre : vecteur des résidus au carré
    """
    xc, yc, R, theta, Dphi, Dlambda = z

    # Calcul des coordonnées sélénographiques
    phi_calc, lambdaa_calc = pxl_vers_selenographique(
        x, y, xc, yc, R, theta, Dphi, Dlambda
    )

    # calcul des résidus
    residus_carre = np.square(
        (lambdaa_cible - lambdaa_calc) * np.cos(phi_cible)
    ) + np.square(phi_calc - phi_cible)
    return residus_carre


def trouver_libration(
    x, y, lambda_cible, phi_cible, xc=0, yc=0, R=1, theta=0, Dphi=0, Dlambda=0
):
    """
    Trouve l'état de libration de la Lune. Les paramètres peuvent être initialisés
    Entrées :
        x, y : 1-d numpy.array, centre des cratères (coordonnées en pixel)
        lambda_cible, phi_cible : 1-d numpy.array, coordonnées sélénographiques correspondantes (longitude, latitude)
        # optionnel
        # xc, yc : centre du disque lunaire (coordonnées en pixel)
        # R : rayon de la Lune (en pixel)
        # theta : inclinaison de l'axe de rotation de Lune par rapport à la vertical (en rad)
        # Dphi, Dlambda : état de libration approché (en rad)
    Sorties :
        Résultat de la fonction least_square de scipy.optimize
    """
    # Initialisation des paramètres
    z0 = np.array([xc, yc, R, theta, Dphi, Dlambda])

    res = optimize.least_squares(
        residus, z0, args=(x, y, lambda_cible, phi_cible), verbose=1
    )
    return res