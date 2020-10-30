import numpy as np


def loadData(filename):
    with open(filename, "rb") as f:
        data = np.loadtxt(
            f, delimiter=";", skiprows=3, usecols=(0, 1, 5, 6, 7), dtype=str
        )
    return data


def getRow(database, key):
    j = 0
    while j < len(database[:, 0]) and database[j, 1] != key:
        j += 1
    if j < len(database[:, 0]):
        return j
    else:
        return -1


def getDim(database, row):
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
    rows = []
    for k in keys:
        row = getRow(database, k)
        if row != -1:
            rows.append(row)
    return rows


def getDims(database, keys):
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
