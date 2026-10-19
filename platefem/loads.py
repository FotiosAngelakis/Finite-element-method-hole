"""Load cases and boundary conditions (load part of the preprocessor).

Edge tractions are converted to *consistent* nodal forces by integrating the
linear shape functions along each boundary edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import PlateData
from .mesh import Mesh

TENSION = "tension"
BENDING = "bending"


@dataclass
class LoadCase:
    kind: str = TENSION  # "tension" or "bending"
    magnitude: float = 2000.0  # tensile force P [N] or bending moment M [N mm]
    # "symmetric": load both ends, restrain the symmetry lines (default)
    # "clamped":   tension only - load at x = 0, clamp the edge x = 2a
    supports: str = "symmetric"

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in (TENSION, BENDING):
            raise ValueError("kind must be 'tension' or 'bending'")
        if self.supports not in ("symmetric", "clamped"):
            raise ValueError("supports must be 'symmetric' or 'clamped'")

    @property
    def code(self) -> int:
        return 1 if self.kind == TENSION else 2


def _edges_on_line(mesh: Mesh, x: float, tol: float = 1e-6) -> np.ndarray:
    edges = mesh.boundary_edges()
    xs = mesh.coords[edges, 0]
    return edges[np.all(np.abs(xs - x) < tol, axis=1)]


def _add_linear_traction(F, mesh, edges, traction, thickness):
    """Consistent nodal forces of an x-traction ``traction(y)`` [N/mm^2]."""
    for p, q in edges:
        yp, yq = mesh.coords[p, 1], mesh.coords[q, 1]
        L = abs(yq - yp)
        tp, tq = traction(yp), traction(yq)
        F[p, 0] += thickness * L / 6 * (2 * tp + tq)
        F[q, 0] += thickness * L / 6 * (tp + 2 * tq)


def apply_load_case(mesh: Mesh, plate: PlateData, load: LoadCase):
    """Return nodal forces ``F`` (n, 2) and boundary conditions ``BC`` (n, 2).

    ``BC[i, j] == 1`` means DOF j (0 = x, 1 = y) of node i is fixed.
    """
    a, b, t = plate.a, plate.b, plate.t
    n = mesh.n_nodes
    F = np.zeros((n, 2))
    BC = np.zeros((n, 2), dtype=int)
    left, right = _edges_on_line(mesh, 0.0), _edges_on_line(mesh, 2 * a)

    if load.kind == TENSION:
        s0 = load.magnitude / (2 * b * t)  # nominal gross-section stress
        _add_linear_traction(F, mesh, left, lambda y: -s0, t)
        if load.supports == "symmetric":
            _add_linear_traction(F, mesh, right, lambda y: +s0, t)
            BC[mesh.nodes_where(x=a), 0] = 1  # vertical symmetry line: ux = 0
            BC[mesh.nodes_where(y=b), 1] = 1  # horizontal symmetry line: uy = 0
        else:  # clamp right edge, uy = 0 at its middle
            BC[mesh.nodes_where(x=2 * a), 0] = 1
            BC[mesh.nodes_where(x=2 * a, y=b), 1] = 1
    else:
        I = (2 * b) ** 3 * t / 12
        M = load.magnitude
        # sigma_x(y) = -M (y - b) / I  (bottom fibre in tension for M > 0)
        _add_linear_traction(F, mesh, left, lambda y: M * (y - b) / I, t)
        _add_linear_traction(F, mesh, right, lambda y: -M * (y - b) / I, t)
        BC[mesh.nodes_where(x=a), 0] = 1
        BC[mesh.nodes_where(x=a, y=0.0), 1] = 1
        BC[mesh.nodes_where(x=a, y=2 * b), 1] = 1
    return F, BC
