"""Mapped triangular meshing of the plate with a central hole.

The plate ``[0, 2a] x [0, 2b]`` with a hole of radius
``R`` centred at ``(a, b)`` is split into four quadrilateral sub-regions.  The
lower-left quarter is meshed with a structured ``noy x nox`` grid of nodes and
then mirrored about ``x = a`` and ``y = b``; nodes on the mirror lines are
shared.  Node numbering: quadrant 1 row by row, then the new nodes of
quadrants 2, 3 and 4.

Lower-left quarter (quadrant 1)::

    (0,b) ----A1---- (a-R,b)
      |                  \\   A2 (quarter arc of the hole)
      |                   \\
      |                    (a, b-R)
      |                       |
    (0,0) ------xds------- (a,0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def spacing(start: float, stop: float, number: int, ratio: float) -> np.ndarray:
    """Geometric point spacing.

    Returns ``number`` points from ``start`` to ``stop`` where every segment is
    ``ratio`` times the previous one (``ratio < 1`` refines towards ``stop``).
    """
    number = int(number)
    if number < 2:
        raise ValueError("spacing needs at least 2 points")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if ratio == 1:
        return np.linspace(start, stop, number)
    first = (stop - start) / ((1 - ratio ** (number - 1)) / (1 - ratio))
    steps = first * ratio ** np.arange(number - 1)
    x = np.empty(number)
    x[0] = start
    x[1:] = start + np.cumsum(steps)
    return x


@dataclass
class Mesh:
    coords: np.ndarray  # (n_nodes, 2) node coordinates
    elements: np.ndarray  # (n_elem, 3) node indices (0-based, counter-clockwise)
    quadrant: np.ndarray  # (n_elem,) sub-region 1..4 each element belongs to
    nox: int = 0
    noy: int = 0

    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def areas(self) -> np.ndarray:
        X = self.coords[self.elements]
        return 0.5 * (
            (X[:, 1, 0] - X[:, 0, 0]) * (X[:, 2, 1] - X[:, 0, 1])
            - (X[:, 2, 0] - X[:, 0, 0]) * (X[:, 1, 1] - X[:, 0, 1])
        )

    @property
    def centroids(self) -> np.ndarray:
        return self.coords[self.elements].mean(axis=1)

    def boundary_edges(self) -> np.ndarray:
        """Edges (node pairs) that belong to exactly one element."""
        e = self.elements
        edges = np.vstack([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]])
        key = np.sort(edges, axis=1)
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return edges[counts[inv.ravel()] == 1]

    def nodes_where(self, x=None, y=None, tol: float = 1e-6) -> np.ndarray:
        mask = np.ones(self.n_nodes, dtype=bool)
        if x is not None:
            mask &= np.abs(self.coords[:, 0] - x) < tol
        if y is not None:
            mask &= np.abs(self.coords[:, 1] - y) < tol
        return np.flatnonzero(mask)

    def nearest_node(self, x: float, y: float) -> int:
        return int(np.argmin(np.hypot(self.coords[:, 0] - x, self.coords[:, 1] - y)))


def _quarter_points(a, b, R, nox, noy, rx, ry):
    """Node grid of the lower-left quarter: arrays (noy, nox) of x and y."""
    n1 = math.ceil(2 * nox / 5)  # points on the straight top part (y = b)
    n2 = math.floor(3 * nox / 5)  # additional points on the hole arc

    xds = spacing(0, a, nox, rx)  # bottom side (y = 0)

    A1 = spacing(0, a - R, n1, rx)  # straight part of the top side
    A2 = spacing(a - R, a - R + math.pi / 2 * R, n2 + 1, rx)  # arc length
    ang = np.cumsum(np.diff(A2)) / R
    ax = np.concatenate([A1, a - R * np.cos(ang)])
    ay = np.concatenate([np.full(n1, b), b - R * np.sin(ang)])

    X = np.empty((noy, nox))
    Y = np.empty((noy, nox))
    for i in range(nox):
        X[:, i] = np.linspace(xds[i], ax[i], noy)
        Y[:, i] = spacing(0, ay[i], noy, ry)
    return X, Y, n1, n2


def _tri_pairs(g: np.ndarray, pattern: int) -> np.ndarray:
    """Split every grid cell of index grid ``g`` into two triangles."""
    out = []
    noy, nox = g.shape
    for j in range(noy - 1):
        for k in range(nox - 1):
            p00, p01 = g[j, k], g[j, k + 1]
            p10, p11 = g[j + 1, k], g[j + 1, k + 1]
            if pattern == 1:  # quadrants 1 and 4  ( /| and |/ )
                out += [(p00, p01, p11), (p00, p11, p10)]
            elif pattern == 2:  # quadrant 2 (mirrored -> reversed orientation)
                out += [(p00, p11, p01), (p00, p10, p11)]
            else:  # quadrant 3
                out += [(p00, p10, p11), (p00, p11, p01)]
    return np.array(out, dtype=int)


def generate_mesh(
    a: float,
    b: float,
    d: float,
    nox: int = 20,
    noy: int = 15,
    ratio_x: float = 0.9,
    ratio_y: float = 1.0,
    decimals: int = 4,
) -> Mesh:
    """Generate the full-plate CST mesh.

    Parameters
    ----------
    a, b : half length / half width of the plate.
    d : hole diameter (hole centred at (a, b)).
    nox, noy : number of nodes per side of the quarter in x / y.
    ratio_x, ratio_y : spacing ratios (< 1 refines towards the hole).
    decimals : node coordinates are rounded to this many decimals.
    """
    nox, noy = int(nox), int(noy)
    R = d / 2
    if nox < 5 or noy < 2:
        raise ValueError("Need nox >= 5 and noy >= 2")
    if not 0 < R < min(a, b):
        raise ValueError("Hole must fit inside the plate")

    X, Y, n1, n2 = _quarter_points(a, b, R, nox, noy, ratio_x, ratio_y)

    pts = []
    # --- quadrant 1 (lower-left) ---------------------------------------
    g1 = np.arange(nox * noy).reshape(noy, nox)
    pts.append(np.column_stack([X.ravel(), Y.ravel()]))
    nid = nox * noy

    # --- quadrant 2: mirror about x = a (shares the column x = a) ------
    X2, Y2 = 2 * a - X[:, : nox - 1], Y[:, : nox - 1]
    g2 = np.empty((noy, nox), dtype=int)
    g2[:, : nox - 1] = nid + np.arange(noy * (nox - 1)).reshape(noy, nox - 1)
    g2[:, nox - 1] = g1[:, nox - 1]
    pts.append(np.column_stack([X2.ravel(), Y2.ravel()]))
    nid += noy * (nox - 1)

    # --- quadrant 3: mirror quadrant 1 about y = b ---------------------
    X3, Y3 = X[: noy - 1, :], 2 * b - Y[: noy - 1, :]
    g3 = np.empty((noy, nox), dtype=int)
    g3[: noy - 1, :] = nid + np.arange((noy - 1) * nox).reshape(noy - 1, nox)
    pts.append(np.column_stack([X3.ravel(), Y3.ravel()]))
    nid += (noy - 1) * nox
    # top row: straight part shared with quadrant 1, arc points mirrored
    g3[noy - 1, :n1] = g1[noy - 1, :n1]
    g3[noy - 1, n1:] = nid + np.arange(n2)
    pts.append(np.column_stack([X[noy - 1, n1:], 2 * b - Y[noy - 1, n1:]]))
    nid += n2

    # --- quadrant 4: mirror quadrant 2 about y = b ---------------------
    X4, Y4 = X2[: noy - 1, :], 2 * b - Y2[: noy - 1, :]
    g4 = np.empty((noy, nox), dtype=int)
    g4[: noy - 1, : nox - 1] = nid + np.arange((noy - 1) * (nox - 1)).reshape(noy - 1, nox - 1)
    g4[:, nox - 1] = g3[:, nox - 1]
    pts.append(np.column_stack([X4.ravel(), Y4.ravel()]))
    nid += (noy - 1) * (nox - 1)
    g4[noy - 1, :n1] = g2[noy - 1, :n1]
    g4[noy - 1, n1 : nox - 1] = nid + np.arange(n2 - 1)
    pts.append(np.column_stack([X2[noy - 1, n1:], 2 * b - Y2[noy - 1, n1:]]))
    nid += n2 - 1

    coords = np.round(np.vstack(pts), decimals)
    assert coords.shape[0] == nid

    elements = np.vstack(
        [_tri_pairs(g1, 1), _tri_pairs(g2, 2), _tri_pairs(g3, 3), _tri_pairs(g4, 1)]
    )
    ne_q = 2 * (nox - 1) * (noy - 1)
    quadrant = np.repeat([1, 2, 3, 4], ne_q)
    mesh = Mesh(coords=coords, elements=elements, quadrant=quadrant, nox=nox, noy=noy)
    if np.any(mesh.areas <= 0):
        raise ValueError(
            "Mesh contains inverted/degenerate elements - try other node counts or ratios"
        )
    return mesh
