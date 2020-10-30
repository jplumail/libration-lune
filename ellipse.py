# Régression elliptique se basant sur la méthode des moindres carrés totaux
# https://fr.wikipedia.org/wiki/R%C3%A9gression_elliptique#M%C3%A9thode_des_moindres_carr%C3%A9s_totaux


from cercle import regression_circulaire

import numpy as np
import scipy.optimize as optimize
from skimage.color import rgb2gray
from skimage import draw
import matplotlib.pyplot as plt


def phis(x, y, cx, cy, theta):
    t = np.arctan2(y - cy, x - cx) - theta  #
    return t


def residus(z, x, y):
    cx, cy, a, b, theta, phis = z[0], z[1], z[2], z[3], z[4], z[5:]
    centre = np.array([[cx], [cy]])
    mat_rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    coords_ellipse = centre + mat_rot @ np.vstack((a * np.cos(phis), b * np.sin(phis)))
    vecteurs_erreur = coords_ellipse - np.vstack((x, y))
    residus = np.linalg.norm(vecteurs_erreur, axis=0)

    return np.square(residus)


def equParametrique_ell(cx, cy, a, b, theta, t):
    centre = np.array([[cx], [cy]])
    mat_rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    coords_ellipse = centre + mat_rot @ np.vstack((a * np.cos(t), b * np.sin(t)))
    return coords_ellipse


def masque_ell(xe, ye, xd, yd, R):
    # Calcule à partir du diamètre a du cratère, de son centre (xe,ye), du centre (xd,yd) et du rayon R du disque lunaire l'inclinaison theta et la
    x_prime = xe - xd
    y_prime = yd - ye
    cos_beta = np.sqrt(R ** 2 - np.square(x_prime) - np.square(y_prime)) / R
    theta = np.arctan2(x_prime, y_prime)
    return theta, cos_beta


def residus2(z, x, y, xd, yd, R):
    cx, cy, a, phis = z[0], z[1], z[2], z[3:]

    theta, cos_beta = masque_ell(cx, cy, xd, yd, R)
    b = a * cos_beta
    centre = np.array([[cx], [cy]])
    mat_rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    coords_ellipse = centre + mat_rot @ np.vstack((a * np.cos(phis), b * np.sin(phis)))
    vecteurs_erreur = coords_ellipse - np.vstack((x, y))
    residus = np.linalg.norm(vecteurs_erreur, axis=0)

    return np.square(residus)


def regression_elliptique(
    x, y, cx=0, cy=0, a=1, b=1, theta=0, directions=None, verbose=0
):
    # Initialisation des paramètres
    N = len(x)
    if directions is None:
        phis = np.zeros((N))
    else:
        phis = directions.copy()
    z0 = np.append([cx, cy, a, b, theta], phis)

    # méthode des moindres carrés
    bounds = (
        [-np.inf] * 4 + [-np.pi] * (N + 1),
        [np.inf] * 4 + [np.pi] * (N + 1),
    )  # je sais pas si cest très utile de restreindre les paramètres...
    res = optimize.least_squares(
        residus, z0, args=(x, y), bounds=bounds, verbose=verbose, max_nfev=50
    )
    return res


def regression_elliptique2(x, y):
    # Dans cette fonction, on initialise les paramètres de la régression elliptique
    # en faisant d'abord une régression circulaire

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
    # xd, yd : centre du disque de la Lune, R : rayon disque
    # Dans cette regression elliptique, la fonction des résidus n'est plus la même
    # residus2 respecte la forme des cratères (voir pdf JLH masque de visée)
    # theta et b ne sont plus des variables :
    # theta (inclinaison de l'ellipse) dépend de la position de l'ellipse sur le disque lunaire : theta est choisi tel que l'ellipse soit dirigée vers le centre du disque
    # b dépend de la distance de l'ellipse au centre du disque : b = a*cos(beta) (voir calcul de cos(beta) dans le pdf de JLH)

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

    # test0 = residus2(z0,x,y,xd,yd,R)
    # print(z0)

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
        print("Problème dans la détection de l'ellipses")
        return None, None, None

    return res, b, theta


"""
from skimage import io

im = io.imread("../images_test/test_ellipse.png")
im = im[:,:,:3]
im2 = rgb2gray(im)
y,x = np.nonzero(im2==0)
yd,xd = 199,329
R = 160

res, b, theta = regression_elliptique3(x,y,xd,yd,R)
x_c,y_c,a = res.x[:3]
phis = res.x[3:]
print(x_c,y_c,a,b,theta*180/np.pi)

coords_ellipse = equParametrique_ell(x_c,y_c,a,b,theta, phis)
print(coords_ellipse.shape)
cc,rr = np.int_(coords_ellipse[0,:]), np.int_(coords_ellipse[1,:])

rr, cc = draw.ellipse_perimeter(int(y_c), int(x_c), int(b), int(a), theta, shape=im.shape)
im[rr,cc] = [0,255,0]

io.imshow(im)
plt.show()"""
