from libration import sellenographique_vers_pxl
from data import getDims

import numpy as np
import matplotlib.pyplot as plt

from skimage import draw


def draw_crateres(
    ax, moon, names, phi, lambdaa, r_crat, xc, yc, R, theta, Dphi, Dlambda
):
    # On affiche un cratère de coordonnées séléno (phi,lambda)
    # Le rayon du cratère est donné en km
    R_lune = 1737
    x, y = sellenographique_vers_pxl(phi, lambdaa, xc, yc, R, theta, Dphi, Dlambda)
    x_prime, y_prime = x - xc, yc - y
    orientation = np.arctan2(x_prime, y_prime)
    x_radius = r_crat * R / R_lune
    cos_beta = np.sqrt(R ** 2 - np.square(x_prime) - np.square(y_prime)) / R
    y_radius = x_radius * cos_beta
    for i in range(len(y)):
        rr, cc = draw.ellipse_perimeter(
            int(y[i]),
            int(x[i]),
            int(y_radius[i]),
            int(x_radius[i]),
            orientation[i],
            shape=moon.shape,
        )
        moon[rr, cc] = [255, 0, 0]
        rr, cc = draw.ellipse_perimeter(
            int(y[i] + 1),
            int(x[i]),
            int(y_radius[i]),
            int(x_radius[i]),
            orientation[i],
            shape=moon.shape,
        )
        moon[rr, cc] = [255, 0, 0]
        rr, cc = draw.ellipse_perimeter(
            int(y[i]),
            int(x[i] + 1),
            int(y_radius[i]),
            int(x_radius[i]),
            orientation[i],
            shape=moon.shape,
        )
        moon[rr, cc] = [255, 0, 0]
        rr, cc = draw.ellipse_perimeter(
            int(y[i] - 1),
            int(x[i]),
            int(y_radius[i]),
            int(x_radius[i]),
            orientation[i],
            shape=moon.shape,
        )
        moon[rr, cc] = [255, 0, 0]
        rr, cc = draw.ellipse_perimeter(
            int(y[i]),
            int(x[i] - 1),
            int(y_radius[i]),
            int(x_radius[i]),
            orientation[i],
            shape=moon.shape,
        )
        moon[rr, cc] = [255, 0, 0]

        ax.text(int(x[i]), int(y[i]), names[i])

    return moon


def draw_meridienEquateur(moon, xc, yc, R, theta, Dphi, Dlambda):
    j = 1 + (min(moon.shape) // 500)
    x_meridien, y_meridien = sellenographique_vers_pxl(
        np.linspace(-np.pi / 2, np.pi / 2, 10000),
        np.zeros((10000)),
        xc,
        yc,
        R,
        theta,
        Dphi,
        Dlambda,
    )
    x_equateur, y_equateur = sellenographique_vers_pxl(
        np.zeros((10000)),
        np.linspace(-np.pi / 2, np.pi / 2, 10000),
        xc,
        yc,
        R,
        theta,
        Dphi,
        Dlambda,
    )
    x_meridien = x_meridien.astype(int)
    y_meridien = y_meridien.astype(int)
    x_equateur = x_equateur.astype(int)
    y_equateur = y_equateur.astype(int)
    cadre_meridien = (
        (x_meridien < (moon.shape[1] - 2 * j))
        & (y_meridien < (moon.shape[0] - 2 * j))
        & (2 * j < x_meridien)
        & (2 * j < y_meridien)
    )
    cadre_equateur = (
        (x_equateur < (moon.shape[1] - 2 * j))
        & (y_equateur < (moon.shape[0] - 2 * j))
        & (2 * j < x_equateur)
        & (2 * j < y_equateur)
    )
    x_meridien = x_meridien[cadre_meridien]
    y_meridien = y_meridien[cadre_meridien]
    x_equateur = x_equateur[cadre_equateur]
    y_equateur = y_equateur[cadre_equateur]
    for i in range(j):
        moon[y_meridien, x_meridien + i] = [255, 0, 0]
        moon[y_meridien, x_meridien - i] = [255, 0, 0]
        moon[y_equateur + i, x_equateur] = [255, 0, 0]
        moon[y_equateur - i, x_equateur] = [255, 0, 0]
    return moon


def draw_resultatsLib(
    ax, moon, x, y, lambdaa_cible, phi_cible, xc, yc, R, theta, Dphi, Dlambda
):
    disque_r, disque_c = draw.circle_perimeter(
        int(yc), int(xc), int(R), shape=moon.shape
    )
    moon[disque_r, disque_c] = [255, 0, 0]
    for i in range(len(x)):
        moon[int(y[i]), int(x[i])] = [255, 0, 0]
        rr, cc = draw.circle_perimeter(int(y[i]), int(x[i]), 10)
        ax.text(int(x[i]), int(y[i]), i + 1)
        moon[rr, cc] = [255, 0, 0]
        x_c, y_c = sellenographique_vers_pxl(
            phi_cible[i], lambdaa_cible[i], xc, yc, R, theta, Dphi, Dlambda
        )
        try:
            moon[int(y_c), int(x_c)] = [0, 255, 0]
        except IndexError:
            print("La calibration a probablement échouée.")

    # plt.rc('text', usetex=True)

    texte = " x = {:.1f} \n y = {:.1f} \n R = {:.1f} \n $\\theta$ = {:.2f}° \n $\Delta \lambda$ = {:.2f}° \n $\Delta \phi$ = {:.2f}°".format(
        xc, yc, R, theta * 180 / np.pi, Dlambda * 180 / np.pi, Dphi * 180 / np.pi
    )
    ax.text(
        0,
        0,
        texte,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="white",
    )

    return moon


def draw_final(ax, moon, database, res, x, y, lon, lat):
    xc, yc, R, theta, Dphi, Dlambda = res.x
    # print("xc, yc, R, theta, Dphi, Dlambda : ",res.x) # avec les angles en radian
    print("Écart-type moyen : ", np.sqrt(res.cost / len(x)) * 180 / np.pi)

    # on dessine les mesures
    moon = draw_resultatsLib(ax, moon, x, y, lon, lat, xc, yc, R, theta, Dphi, Dlambda)

    # On dessine des cratères
    crateres = [
        "AA5361N05648E",
        "AA5361N05648E",
        "AA5361N05648E",
        "AA5361N05648E",
        "AA5162N00938W",
        "AA5024N01732E",
        "SF0321S00519W",
        "AA1626N01593E",
        "AA1626N01593E",
        "AA0886S06104E",
        "AA2539S06078E",
        "AA7773N01413E",
        "AA6952N05123W",
        "AA1445N00907E",
        "AA0962N02008W",
        "AA4330S01122W",
        "AA2373N04749W",
    ]
    names, lat_crat, lon_crat, rad_crat = getDims(database, crateres)
    moon = draw_crateres(
        ax, moon, names, lat_crat, lon_crat, rad_crat, xc, yc, R, theta, Dphi, Dlambda
    )
    moon = draw_meridienEquateur(moon, xc, yc, R, theta, Dphi, Dlambda)

    return moon
