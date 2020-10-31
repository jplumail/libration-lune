"""
Contient les fonctions permettant d'accéder à la base de données des cratères
"""

import numpy as np


def loadData(filename):
    """
    Charge la base de données
    Entrée : emplacement de la base de données
    Sortie : numpy.array contenant la base de données
    """
    with open(filename, "rb") as f:
        data = np.loadtxt(
            f, delimiter=";", skiprows=3, usecols=(0, 1, 5, 6, 7), dtype=str
        )
    return data


def getRow(database, key):
    """
    Renvoie le l'index du cratère correspondant
    Entrées :
        - database : base de données (sortie de loadData)
        - key : code UAI du cratère
    Sortie : index du cratère dans la base de données OU -1 si le cratère est introuvable
    """
    j = 0
    while j < len(database[:, 0]) and database[j, 1] != key:
        j += 1
    if j < len(database[:, 0]):
        return j
    else:
        return -1


def getDim(database, row):
    """
    Renvoie les dimensions d'un cratère
    Entrées :
        - database : base de données (sortie de loadData)
        - row : index du cratère dans la base de données
    Sorties :
        - nom du cratère
        - latitude sélénographique du cratère
        - longitude sélénographique du cratère
        - rayon du cratère
    """
    name, diam, lat, lon = (
        database[row, 0],
        float(database[row, 2]),
        float(database[row, 3]),
        float(database[row, 4]),
    )
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    radius = diam / 2
    return name, lat, lon, radius


def getRows(database, keys):
    """
    Même fonction que getRow pour plusieurs cratères
    Entrées :
        - database : base de données (sortie de loadData)
        - keys : liste de codes UAI de cratères
    Sortie : liste d'index des cratères
    """
    rows = []
    for k in keys:
        row = getRow(database, k)
        rows.append(row)
    return rows


def getDims(database, keys):
    """
    Même fonction que getDims pour plusieurs cratères
    Entrées :
        - database : base de données (sortie de loadData)
        - keys : liste de codes UAI de cratères
    Sorties :
        - liste des noms des cratères
        - liste des latitudes sélénographiques des cratères
        - liste des longitudes sélénographiques des cratères
        - liste des rayons des cratères
    """
    rows = getRows(database, keys)
    names = []
    lat = []
    lon = []
    radius = []
    for r in rows:
        dim = getDim(database, r)
        names.append(dim[0])
        lat.append(dim[1])
        lon.append(dim[2])
        radius.append(dim[3])
    return np.array(names), np.array(lat), np.array(lon), np.array(radius)
