"""Post-processing: nodal stress recovery and notch (stress-concentration) factors.

CST elements give
one constant stress per element.  To colour the plate continuously and to get a
better estimate of the peak stress at the hole, element values are

* averaged to the nodes (area-weighted) everywhere, and
* recovered with a least-squares patch fit on the vertical symmetry line
  ``x = a`` (the net section where the peak occurs), which extrapolates the
  steep stress gradient at the hole much better than plain averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import PlateData
from .loads import TENSION, LoadCase
from .mesh import Mesh
from .solver import Results, von_mises


# --------------------------------------------------------------------------
# Theory (Peterson's stress concentration factors)
# --------------------------------------------------------------------------
def kt_tension_theory(d_over_H: float) -> float:
    """Finite-width plate with a central hole in tension, net-section Ktn
    (Peterson / Heywood fit, 0 <= d/H <= 1)."""
    r = d_over_H
    return 3.0 - 3.14 * r + 3.667 * r**2 - 1.527 * r**3


def kt_bending_theory(d_over_H: float) -> float:
    """Plate with a central hole under in-plane bending, Ktn based on
    sigma_nom = 6 M d / ((H^3 - d^3) t). Peterson's chart gives Ktn ~= 2 for
    d/H up to about 0.65."""
    return 2.0


def nominal_stress(plate: PlateData, load: LoadCase) -> float:
    H, d, t = plate.H, plate.d, plate.t
    if load.kind == TENSION:
        return abs(load.magnitude) / (t * (H - d))
    return 6 * abs(load.magnitude) * d / ((H**3 - d**3) * t)


def edge_stress_theory(plate: PlateData, load: LoadCase) -> float:
    """Bending: stress at the plate edge (y = 0) of the net section x = a."""
    H, d, t = plate.H, plate.d, plate.t
    return 6 * abs(load.magnitude) * H / ((H**3 - d**3) * t)


# --------------------------------------------------------------------------
# Nodal recovery
# --------------------------------------------------------------------------
def nodal_average(mesh: Mesh, values: np.ndarray) -> np.ndarray:
    """Area-weighted average of element values at the nodes."""
    values = np.asarray(values, dtype=float)
    w = mesh.areas
    vals = values.reshape(len(values), -1)
    acc = np.zeros((mesh.n_nodes, vals.shape[1]))
    wsum = np.zeros(mesh.n_nodes)
    for k in range(3):
        np.add.at(acc, mesh.elements[:, k], vals * w[:, None])
        np.add.at(wsum, mesh.elements[:, k], w)
    out = acc / wsum[:, None]
    return out.reshape((mesh.n_nodes,) + values.shape[1:])


def _element_patch(mesh: Mesh, node: int, rings: int) -> np.ndarray:
    """Boolean mask of the elements within ``rings`` element layers of ``node``."""
    nodes = np.array([node])
    mask = np.any(np.isin(mesh.elements, nodes), axis=1)
    for _ in range(rings):
        nodes = np.unique(mesh.elements[mask])
        mask = np.any(np.isin(mesh.elements, nodes), axis=1)
    return mask


def patch_recovery(
    mesh: Mesh, values: np.ndarray, nodes, rings: int = 1, degree: int = 2
) -> np.ndarray:
    """Least-squares patch recovery of element values at selected nodes.

    For every node, the element (centroid) values of a patch of elements around
    it are fitted with a polynomial in (x, y) and the fit is evaluated at the
    node.  This extrapolates the steep gradients at the hole much better than
    plain averaging (Zienkiewicz-Zhu style recovery).
    """
    values = np.asarray(values, dtype=float)
    vals2 = values.reshape(len(values), -1)
    cg = mesh.centroids
    out = np.empty((len(nodes), vals2.shape[1]))
    for i, n in enumerate(nodes):
        el = _element_patch(mesh, int(n), rings)
        dx = cg[el] - mesh.coords[n]
        cols = [np.ones(el.sum()), dx[:, 0], dx[:, 1]]
        if degree == 2 and el.sum() >= 8:
            cols += [dx[:, 0] ** 2, dx[:, 0] * dx[:, 1], dx[:, 1] ** 2]
        A = np.column_stack(cols)
        coef = np.linalg.lstsq(A, vals2[el], rcond=None)[0]
        out[i] = coef[0]
    return out.reshape((len(nodes),) + values.shape[1:])


@dataclass
class LineStress:
    """sigma_x along the vertical symmetry line x = a (the net section)."""

    nodes: np.ndarray  # node indices, sorted by y
    y: np.ndarray  # node y-coordinates
    sx: np.ndarray  # recovered nodal sigma_x


def symmetry_line_stress(mesh: Mesh, plate: PlateData, sx: np.ndarray) -> LineStress:
    """Recover sigma_x at the nodes of the net section x = a."""
    nodes = mesh.nodes_where(x=plate.a)
    nodes = nodes[np.argsort(mesh.coords[nodes, 1])]
    return LineStress(nodes, mesh.coords[nodes, 1], patch_recovery(mesh, sx, nodes))


@dataclass
class NodalResults:
    strains: np.ndarray  # (n, 3)
    stresses: np.ndarray  # (n, 3)
    von_mises: np.ndarray  # (n,)
    line: LineStress


def recover_nodal(mesh: Mesh, plate: PlateData, res: Results) -> NodalResults:
    """Nodal strains/stresses: averaging + patch recovery on the line x = a."""
    eps = nodal_average(mesh, res.strains)
    sig = nodal_average(mesh, res.stresses)
    line = symmetry_line_stress(mesh, plate, res.stresses[:, 0])
    sig[line.nodes] = patch_recovery(mesh, res.stresses, line.nodes)
    eps[line.nodes] = patch_recovery(mesh, res.strains, line.nodes)
    return NodalResults(eps, sig, von_mises(sig), line)


# --------------------------------------------------------------------------
# Notch factor
# --------------------------------------------------------------------------
@dataclass
class NotchResult:
    s_nominal: float
    s_max_element: float
    s_max_nodal: float
    kt_element: float
    kt_nodal: float
    kt_theory: float
    s_max_theory: float
    edge_stress: float | None = None  # bending only (nodal, at (a, 0))
    edge_stress_theory: float | None = None

    def summary(self) -> str:
        lines = [
            f"Nominal stress           : {self.s_nominal:10.4f} MPa",
            f"Max stress (element)     : {self.s_max_element:10.4f} MPa  ->  Kt = {self.kt_element:.4f}",
            f"Max stress (nodal/patch) : {self.s_max_nodal:10.4f} MPa  ->  Kt = {self.kt_nodal:.4f}",
            f"Theory (Peterson)        : {self.s_max_theory:10.4f} MPa  ->  Kt = {self.kt_theory:.4f}",
        ]
        if self.edge_stress is not None:
            lines.append(
                f"Edge stress at x=a, y=0  : {self.edge_stress:10.4f} MPa  (theory {self.edge_stress_theory:.4f})"
            )
        return "\n".join(lines)


def notch_factor(
    mesh: Mesh, plate: PlateData, load: LoadCase, res: Results, nodal: NodalResults | None = None
) -> NotchResult:
    a, b, R = plate.a, plate.b, plate.R
    sx = res.stresses[:, 0]
    if nodal is None:
        nodal = recover_nodal(mesh, plate, res)
    s_nom = nominal_stress(plate, load)
    dH = plate.d / plate.H
    hole_node = mesh.nearest_node(a, b - R)  # bottom of the hole on x = a

    if load.kind == TENSION:
        kt_th = kt_tension_theory(dH)
        s_el = float(np.max(np.abs(sx)))
        s_nd = float(abs(nodal.stresses[hole_node, 0]))
        edge, edge_th = None, None
    else:
        kt_th = kt_bending_theory(dH)
        touching = np.any(mesh.elements == hole_node, axis=1)
        s_el = float(np.max(np.abs(sx[touching])))
        s_nd = float(abs(nodal.stresses[hole_node, 0]))
        edge_node = mesh.nearest_node(a, 0.0)
        edge = float(abs(nodal.stresses[edge_node, 0]))
        edge_th = edge_stress_theory(plate, load)

    return NotchResult(
        s_nominal=s_nom,
        s_max_element=s_el,
        s_max_nodal=s_nd,
        kt_element=s_el / s_nom,
        kt_nodal=s_nd / s_nom,
        kt_theory=kt_th,
        s_max_theory=kt_th * s_nom,
        edge_stress=edge,
        edge_stress_theory=edge_th,
    )
