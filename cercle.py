# Régression circulaire avec la méthode des moindres carrés totaux
# https://fr.wikipedia.org/wiki/R%C3%A9gression_circulaire

import numpy as np
import scipy.optimize as optimize


def residus(c, x, y):
    r = radius(c, x, y)
    N = len(x)
    diff = np.zeros((2, N))
    diff[0, :] = x - c[0]
    diff[1, :] = y - c[1]
    norm_diff = np.linalg.norm(diff, axis=0)
    return np.square(norm_diff - r)


def radius(c, x, y):
    N = len(x)
    diff = np.zeros((2, N))
    diff[0, :] = x - c[0]
    diff[1, :] = y - c[1]
    r = np.sum(np.linalg.norm(diff, axis=0)) / N
    return r


def regression_circulaire(c0, x, y):
    res = optimize.least_squares(residus, c0, args=(x, y))
    r = radius(res.x, x, y)
    return res, r
