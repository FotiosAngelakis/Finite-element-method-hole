"""Linear CST solver.

Builds the element stiffness matrices ``Ke = t A B^T D B`` for constant-strain
triangles, assembles the global (sparse) stiffness matrix, applies the
boundary conditions and computes displacements, reactions, strains, stresses,
principal stresses and the von Mises equivalent stress per element.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import Mesh


def constitutive_matrix(E: float, v: float) -> np.ndarray:
    """Plane-stress elasticity matrix D."""
    return E / (1 - v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])


def strain_displacement(coords: np.ndarray, elements: np.ndarray):
    """B matrices (m, 3, 6) and areas (m,) of all CST elements."""
    X = coords[elements]
    x, y = X[..., 0], X[..., 1]
    bb = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    cc = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    A = 0.5 * (bb[:, 0] * cc[:, 1] - bb[:, 1] * cc[:, 0])
    B = np.zeros((len(elements), 3, 6))
    B[:, 0, 0::2] = bb
    B[:, 1, 1::2] = cc
    B[:, 2, 0::2] = cc
    B[:, 2, 1::2] = bb
    B /= (2 * A)[:, None, None]
    return B, A


def element_dofs(elements: np.ndarray) -> np.ndarray:
    dofs = np.empty((len(elements), 6), dtype=int)
    dofs[:, 0::2] = 2 * elements
    dofs[:, 1::2] = 2 * elements + 1
    return dofs


def assemble_stiffness(mesh: Mesh, E: float, v: float, t) -> sp.csr_matrix:
    B, A = strain_displacement(mesh.coords, mesh.elements)
    D = constitutive_matrix(E, v)
    t = np.broadcast_to(np.asarray(t, dtype=float), A.shape)
    Ke = np.einsum("eji,jk,ekl->eil", B, D, B) * (t * A)[:, None, None]
    dofs = element_dofs(mesh.elements)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    n = 2 * mesh.n_nodes
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


@dataclass
class Results:
    U: np.ndarray  # (n, 2) nodal displacements
    reactions: np.ndarray  # (n, 2) support reactions (K u - F)
    strains: np.ndarray  # (m, 3) ex, ey, gxy
    stresses: np.ndarray  # (m, 3) sx, sy, txy
    principal: np.ndarray  # (m, 3) s1, s2, phi (angle of s1 to x, rad)
    von_mises: np.ndarray  # (m,)
    new_coords: np.ndarray  # (n, 2) deformed node coordinates
    cg_deformed: np.ndarray  # (m, 2) deformed element centroids

    @property
    def U_total(self) -> np.ndarray:
        return np.hypot(self.U[:, 0], self.U[:, 1])


def principal_stresses(s: np.ndarray) -> np.ndarray:
    sx, sy, txy = s[:, 0], s[:, 1], s[:, 2]
    c = (sx + sy) / 2
    r = np.sqrt(((sx - sy) / 2) ** 2 + txy**2)
    phi = 0.5 * np.arctan2(2 * txy, sx - sy)  # direction of s1
    return np.column_stack([c + r, c - r, phi])


def von_mises(s: np.ndarray) -> np.ndarray:
    sx, sy, txy = s[:, 0], s[:, 1], s[:, 2]
    return np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2)


def solve(
    mesh: Mesh,
    F: np.ndarray,
    BC: np.ndarray,
    E: float,
    v: float,
    t=4.0,
    method: str = "elimination",
    penalty: float = 1e7,
) -> Results:
    """Solve K u = F.

    ``method="elimination"`` removes the fixed DOFs exactly (default).
    ``method="penalty"`` adds ``penalty`` to the diagonal of every fixed DOF.
    """
    K = assemble_stiffness(mesh, E, v, t)
    f = np.asarray(F, dtype=float).ravel()
    fixed = np.asarray(BC, dtype=bool).ravel()
    n = f.size
    u = np.zeros(n)
    if method == "penalty":
        Kp = K + sp.diags(np.where(fixed, penalty, 0.0))
        u = spla.spsolve(Kp.tocsc(), f)
    elif method == "elimination":
        free = ~fixed
        Kff = K[free][:, free].tocsc()
        msg = "Stiffness matrix is singular - the supports do not prevent rigid-body motion"
        try:
            lu = spla.splu(Kff)
        except RuntimeError as exc:
            raise RuntimeError(msg) from exc
        piv = np.abs(lu.U.diagonal())
        if piv.size == 0 or piv.min() < 1e-10 * piv.max():
            raise RuntimeError(msg)
        u[free] = lu.solve(f[free])
    else:
        raise ValueError("method must be 'elimination' or 'penalty'")
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Solution is not finite - check boundary conditions")

    U = u.reshape(-1, 2)
    reactions = (K @ u - f).reshape(-1, 2)

    B, _ = strain_displacement(mesh.coords, mesh.elements)
    ue = u[element_dofs(mesh.elements)]
    strains = np.einsum("eij,ej->ei", B, ue)
    stresses = strains @ constitutive_matrix(E, v).T
    new_coords = mesh.coords + U
    return Results(
        U=U,
        reactions=reactions,
        strains=strains,
        stresses=stresses,
        principal=principal_stresses(stresses),
        von_mises=von_mises(stresses),
        new_coords=new_coords,
        cg_deformed=new_coords[mesh.elements].mean(axis=1),
    )
