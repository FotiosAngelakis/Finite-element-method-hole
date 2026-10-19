"""Text-file exchange between the three stages.

Preprocessor  -> <prefix>_Nodes_Plate.txt, <prefix>_Elements_Plate.txt, <prefix>_data_Plate.txt
Solver        -> + _Displacement_Plate.txt, _Stresses_Plate.txt, _Strains_Plate.txt,
                   _Reactions_Plate.txt (and updates Nodes/Elements with deformed data)

All files are comma separated with one header line; node/element ids are
written 1-based.  A compact data file containing only ``E, v, G, a, b`` is
also accepted (thickness then comes from the elements file and the hole
diameter from the node coordinates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import PlateData
from .loads import LoadCase
from .mesh import Mesh
from .solver import Results

FILES = {
    "nodes": "Nodes_Plate.txt",
    "elements": "Elements_Plate.txt",
    "data": "data_Plate.txt",
    "disp": "Displacement_Plate.txt",
    "stress": "Stresses_Plate.txt",
    "strain": "Strains_Plate.txt",
    "react": "Reactions_Plate.txt",
}


def path(folder, prefix: str, key: str) -> Path:
    return Path(folder) / f"{prefix}_{FILES[key]}"


def _write(fname: Path, header: list[str], data: np.ndarray, fmts: list[str]) -> None:
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write(",".join(f"{h:>14s}" for h in header) + "\n")
        for row in data:
            fh.write(",".join(f % v for f, v in zip(fmts, row)) + "\n")


def _read(fname: Path) -> np.ndarray:
    if not Path(fname).exists():
        raise FileNotFoundError(f"{fname} not found")
    arr = np.genfromtxt(fname, delimiter=",", skip_header=1)
    return np.atleast_2d(arr)


# ------------------------------------------------------------------ write
def write_preprocessor(folder, prefix, mesh: Mesh, plate: PlateData, load: LoadCase, F, BC):
    n = mesh.n_nodes
    nodes = np.column_stack([np.arange(1, n + 1), mesh.coords, BC, F])
    _write(
        path(folder, prefix, "nodes"),
        ["id", "x", "y", "BCx", "BCy", "Fx", "Fy"],
        nodes,
        ["%14d", "%14.6f", "%14.6f", "%14d", "%14d", "%14.6e", "%14.6e"],
    )
    m = mesh.n_elements
    t = np.full(m, plate.t)
    els = np.column_stack(
        [np.arange(1, m + 1), mesh.elements + 1, mesh.areas, mesh.centroids, t, mesh.quadrant]
    )
    _write(
        path(folder, prefix, "elements"),
        ["id", "N1", "N2", "N3", "A", "CGx", "CGy", "t", "quadrant"],
        els,
        ["%14d"] * 4 + ["%14.6e", "%14.6f", "%14.6f", "%14.4f", "%14d"],
    )
    data = np.array(
        [[plate.E, plate.v, plate.G, plate.a, plate.b, plate.t, plate.d,
          load.code, load.magnitude, mesh.nox, mesh.noy]]
    )
    _write(
        path(folder, prefix, "data"),
        ["E", "v", "G", "a", "b", "t", "d", "load(1=T,2=B)", "P_or_M", "nox", "noy"],
        data,
        ["%14.6g"] * 11,
    )


def write_solver(folder, prefix, mesh: Mesh, plate: PlateData, F, BC, res: Results):
    n = mesh.n_nodes
    nodes = np.column_stack([np.arange(1, n + 1), mesh.coords, BC, F, res.new_coords])
    _write(
        path(folder, prefix, "nodes"),
        ["id", "x", "y", "BCx", "BCy", "Fx", "Fy", "xnew", "ynew"],
        nodes,
        ["%14d", "%14.6f", "%14.6f", "%14d", "%14d", "%14.6e", "%14.6e", "%14.8f", "%14.8f"],
    )
    m = mesh.n_elements
    els = np.column_stack(
        [np.arange(1, m + 1), mesh.elements + 1, mesh.areas, mesh.centroids,
         np.full(m, plate.t), res.cg_deformed, mesh.quadrant]
    )
    _write(
        path(folder, prefix, "elements"),
        ["id", "N1", "N2", "N3", "A", "CGx", "CGy", "t", "CGdefx", "CGdefy", "quadrant"],
        els,
        ["%14d"] * 4 + ["%14.6e", "%14.6f", "%14.6f", "%14.4f", "%14.8f", "%14.8f", "%14d"],
    )
    e = "%14.6e"
    _write(path(folder, prefix, "disp"), ["Ux", "Uy"], res.U, [e, e])
    _write(path(folder, prefix, "react"), ["Rx", "Ry"], res.reactions, [e, e])
    _write(path(folder, prefix, "strain"), ["ex", "ey", "gxy"], res.strains, [e] * 3)
    _write(
        path(folder, prefix, "stress"),
        ["sx", "sy", "txy", "s1", "s2", "phi", "VonMises"],
        np.column_stack([res.stresses, res.principal, res.von_mises]),
        [e] * 7,
    )


# ------------------------------------------------------------------- read
def read_model(folder, prefix):
    """Read the preprocessor files -> (mesh, plate, load, F, BC)."""
    N = _read(path(folder, prefix, "nodes"))
    E = _read(path(folder, prefix, "elements"))
    D = _read(path(folder, prefix, "data"))[0]
    coords = N[:, 1:3]
    BC = N[:, 3:5].astype(int)
    F = N[:, 5:7]
    elements = E[:, 1:4].astype(int) - 1
    ne = len(elements)
    if E.shape[1] in (9, 11):  # platefem format: quadrant is the last column
        quadrant = E[:, -1].astype(int)
    else:
        quadrant = np.repeat([1, 2, 3, 4], ne // 4) if ne % 4 == 0 else np.ones(ne, int)
    thickness = float(E[0, 7])
    E_mod, v, _G, a, b = D[:5]
    if len(D) >= 11:  # file written by platefem
        t, d = D[5], D[6]
        load = LoadCase("tension" if int(D[7]) == 1 else "bending", float(D[8]))
        nox, noy = int(D[9]), int(D[10])
    else:  # compact data file (E, v, G, a, b only)
        t = thickness
        r = np.hypot(coords[:, 0] - a, coords[:, 1] - b)
        d = 2 * float(r.min())
        load = LoadCase("tension", float(abs(F[:, 0].sum() / 2)) or 1.0)
        nox = noy = 0
    plate = PlateData(t=t, d=round(d, 4), E=E_mod, v=v, a=a, b=b)
    mesh = Mesh(coords=coords, elements=elements, quadrant=quadrant, nox=nox, noy=noy)
    return mesh, plate, load, F, BC


def read_results(folder, prefix, mesh: Mesh) -> Results:
    """Read the solver output files back into a :class:`Results` object."""
    U = _read(path(folder, prefix, "disp"))
    R = _read(path(folder, prefix, "react"))
    eps = _read(path(folder, prefix, "strain"))
    S = _read(path(folder, prefix, "stress"))
    new = mesh.coords + U
    return Results(
        U=U,
        reactions=R,
        strains=eps[:, :3],
        stresses=S[:, :3],
        principal=S[:, 3:6],
        von_mises=S[:, 6],
        new_coords=new,
        cg_deformed=new[mesh.elements].mean(axis=1),
    )
