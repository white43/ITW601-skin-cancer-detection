import math

import numpy as np


def area(polygon: list[tuple[int, int]]) -> float:
    """
    https://mathworld.wolfram.com/PolygonArea.html

    Parameters
    ----------
    polygon: list[tuple[int, int]]
        A list of vertices of a polygon. Each vertex is a tuple of (x, y) coordinates.
    Returns
    -------
    float
        Unsigned area
    """
    s: int = 0
    x, y = 0, 1
    pl = len(polygon)

    if pl < 3:
        return 0

    for i in range(pl):
        c = polygon[i]
        n = polygon[i + 1] if i + 1 < len(polygon) else polygon[0]

        s += (c[x] * n[y] - n[x] * c[y])

    return abs(s / 2)


def perimeter(polygon: list[tuple[int, int]]) -> float:
    s = 0
    x, y = 0, 1
    pl = len(polygon)

    if pl < 2:
        return 0

    for i in range(pl):
        c = polygon[i]
        n = polygon[i + 1] if i + 1 < len(polygon) else polygon[0]

        s += math.sqrt((n[x] - c[x]) ** 2 + (n[y] - c[y]) ** 2)

    return s


def centroid(polygon: list[tuple[int, int]]) -> tuple[float, float]:
    """
    https://mathworld.wolfram.com/PolygonCentroid.html

    Parameters
    ----------
    polygon: list[tuple[int, int]]
        A list of vertices of a polygon. Each vertex is a tuple of (x, y) coordinates.
    Returns
    -------
    tuple[float, float]
        Coordinates of centroid
    """
    xs, ys = 0, 0
    x, y = 0, 1
    pl = len(polygon)

    if pl < 3:
        return 0, 0

    for i in range(pl):
        c = polygon[i]
        n = polygon[i + 1] if i + 1 < len(polygon) else polygon[0]

        xs += (c[x] + n[x]) * (c[x] * n[y] - n[x] * c[y])
        ys += (c[y] + n[y]) * (c[x] * n[y] - n[x] * c[y])

    s = area(polygon)

    xs /= 6 * s
    ys /= 6 * s

    return abs(xs), abs(ys)


def circularity(polygon: list[tuple[int, int]]) -> float:
    """
    https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)#Circularity
    Parameters
    ----------
    polygon: list[tuple[int, int]]
        A list of vertices of a polygon. Each vertex is a tuple of (x, y) coordinates.
    Returns
    -------
    float
        Measure of polygon's circularity where 1 denotes a circle and values less than 1 denote less circular
        shapes
    """
    if len(polygon) < 3:
        return 0

    a = area(polygon)
    p = perimeter(polygon)

    return 4 * math.pi * a / p ** 2


def shape_irregularity(polygon: list[tuple[int, int]]) -> float:
    x, y = 0, 1

    if len(polygon) < 3:
        return 0

    c = centroid(polygon)

    lens = np.array([math.sqrt((v[x] - c[x]) ** 2 + (v[y] - c[y]) ** 2) for v in polygon])

    mean = lens.mean()
    stddev = lens.std()

    return stddev / mean
