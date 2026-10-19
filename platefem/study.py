"""Mesh sensitivity studies.

1. Convergence of the notch factor / peak stress with mesh density, refining
   both directions together and each direction separately.
2. Distribution of sigma_x on the vertical symmetry axis for several meshes.
3. Bending: hole diameter for which the notch stress equals the edge stress.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from . import plots
from .data import PlateData
from .loads import LoadCase, apply_load_case
from .mesh import generate_mesh
from .postprocess import (edge_stress_theory, kt_bending_theory, nominal_stress,
                          notch_factor, recover_nodal)
from .solver import solve

DEFAULT_MAGNITUDE = {"tension": 2000.0, "bending": 10000.0}


def graded_ratio(n: int, base_ratio=0.9, base_n=20) -> float:
    """Ratio giving the same overall grading r^(n-1) as ``base_ratio`` with ``base_n`` nodes."""
    return base_ratio ** ((base_n - 1) / (n - 1))


def analyse(plate: PlateData, load: LoadCase, nox, noy, rx=None, ry=1.0):
    rx = graded_ratio(nox) if rx is None else rx
    # strong grading can fold elements for some hole sizes -> relax the ratio
    for r in (rx, rx**0.5, rx**0.25, 1.0):
        try:
            mesh = generate_mesh(plate.a, plate.b, plate.d, nox, noy, r, ry)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Could not mesh d={plate.d} with {nox}x{noy} nodes")
    F, BC = apply_load_case(mesh, plate, load)
    res = solve(mesh, F, BC, plate.E, plate.v, plate.t)
    nodal = recover_nodal(mesh, plate, res)
    return mesh, res, nodal, notch_factor(mesh, plate, load, res, nodal)


def _write_csv(fname: Path, header, rows):
    with open(fname, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------ convergence
SERIES = {
    "both x and y": [(8, 6), (10, 8), (14, 10), (20, 15), (26, 19), (32, 24), (40, 30), (50, 38)],
    "x only (noy = 15)": [(8, 15), (10, 15), (14, 15), (20, 15), (26, 15), (32, 15), (40, 15), (50, 15)],
    "y only (nox = 20)": [(20, 4), (20, 6), (20, 8), (20, 12), (20, 15), (20, 20), (20, 30), (20, 40)],
}


def convergence(kind: str, plate: PlateData, folder: Path, magnitude=None, series=None):
    series = SERIES if series is None else series
    load = LoadCase(kind, DEFAULT_MAGNITUDE[kind] if magnitude is None else magnitude)
    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, meshes in series.items():
        ne, kte, ktn = [], [], []
        for nox, noy in meshes:
            mesh, res, nodal, n = analyse(plate, load, nox, noy)
            ne.append(mesh.n_elements)
            kte.append(n.kt_element)
            ktn.append(n.kt_nodal)
            rows.append([name, nox, noy, mesh.n_elements, n.s_max_element, n.s_max_nodal,
                         n.kt_element, n.kt_nodal, n.kt_theory])
        l, = ax1.plot(ne, ktn, "o-", label=f"{name} - nodal (recovered)")
        ax1.plot(ne, kte, "s--", color=l.get_color(), alpha=0.6, label=f"{name} - element")
        ax2.plot(ne, np.array(ktn) * n.s_nominal, "o-", label=name)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("Number of elements")
        ax.grid(alpha=0.3, which="both")
    ax1.axhline(n.kt_theory, color="k", ls="--", label=f"Theory Kt = {n.kt_theory:.3f}")
    ax1.set_ylabel("Notch factor Kt")
    ax1.set_title(f"{kind.capitalize()}: notch factor convergence")
    ax1.legend(fontsize=7)
    ax2.axhline(n.s_max_theory, color="k", ls="--", label=f"Theory {n.s_max_theory:.3f} MPa")
    ax2.set_ylabel("Max $\\sigma_x$ at the hole [MPa]")
    ax2.set_title(f"{kind.capitalize()}: peak stress convergence")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / f"convergence_{kind}.png", dpi=150)
    plt.close(fig)
    _write_csv(folder / f"convergence_{kind}.csv",
               ["series", "nox", "noy", "elements", "smax_element", "smax_nodal",
                "kt_element", "kt_nodal", "kt_theory"], rows)
    return rows


# ------------------------------------------------- symmetry-line distribution
def line_distribution(kind: str, plate: PlateData, folder: Path, magnitude=None,
                      meshes=((8, 6), (14, 10), (20, 15), (32, 24), (50, 38))):
    load = LoadCase(kind, DEFAULT_MAGNITUDE[kind] if magnitude is None else magnitude)
    curves, rows = [], []
    for nox, noy in meshes:
        mesh, res, nodal, n = analyse(plate, load, nox, noy)
        curves.append((f"{nox}x{noy} ({mesh.n_elements} el.)", nodal.line.y, nodal.line.sx))
        rows += [[nox, noy, y, s] for y, s in zip(nodal.line.y, nodal.line.sx)]
    fig = plots.plot_line_distribution(
        curves, title=f"{kind.capitalize()}: $\\sigma_x$ on the symmetry axis x = a",
        theory=n.s_max_theory)
    ax = fig.axes[0]
    if kind == "bending":
        y = np.linspace(0, plate.H, 200)
        I_net = plate.t * (plate.H**3 - plate.d**3) / 12
        s_beam = -load.magnitude * (y - plate.b) / I_net
        s_beam[np.abs(y - plate.b) < plate.R] = np.nan
        ax.plot(s_beam, y, "k:", lw=1, label="Beam theory (net section)")
        ax.axvline(-n.s_max_theory, color="k", ls="--", lw=1)
    else:
        ax.axvline(n.s_nominal, color="0.4", ls=":", lw=1, label="Nominal (net) stress")
    ax.axhspan(plate.b - plate.R, plate.b + plate.R, color="0.9", label="Hole")
    ax.legend(fontsize=8)
    fig.savefig(folder / f"line_distribution_{kind}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _write_csv(folder / f"line_distribution_{kind}.csv", ["nox", "noy", "y", "sx"], rows)


# ------------------------------------------------------------ d search
def diameter_search(plate: PlateData, folder: Path, magnitude=None, nox=40, noy=30,
                    d_values=None):
    """Bending: find d with notch stress (hole, x = a) == edge stress (x = a, y = 0)."""
    load = LoadCase("bending", DEFAULT_MAGNITUDE["bending"] if magnitude is None else magnitude)
    d_values = np.arange(30.0, 72.5, 2.5) if d_values is None else np.asarray(d_values)
    notch, edge, notch_th, edge_th = [], [], [], []
    for d in d_values:
        p = PlateData(t=plate.t, d=float(d), E=plate.E, v=plate.v, a=plate.a, b=plate.b)
        _, _, _, n = analyse(p, load, nox, noy)
        notch.append(n.s_max_nodal)
        edge.append(n.edge_stress)
        notch_th.append(kt_bending_theory(d / p.H) * nominal_stress(p, load))
        edge_th.append(edge_stress_theory(p, load))
    notch, edge = np.array(notch), np.array(edge)
    diff = notch - edge
    d_fem = np.nan
    idx = np.flatnonzero(np.sign(diff[:-1]) != np.sign(diff[1:]))
    if len(idx):
        i = idx[0]
        d_fem = d_values[i] - diff[i] * (d_values[i + 1] - d_values[i]) / (diff[i + 1] - diff[i])
    H = plate.H
    d_th = brentq(lambda d: kt_bending_theory(d / H) * d - H, 1e-3, H - 1e-3)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(d_values, notch, "o-", label="FEM notch stress (hole, x = a)")
    ax.plot(d_values, edge, "s-", label="FEM edge stress (x = a, y = 0)")
    ax.plot(d_values, notch_th, "--", color="C0", alpha=0.7, label="Theory notch: Kt $\\sigma_{nom}$")
    ax.plot(d_values, edge_th, "--", color="C1", alpha=0.7, label="Beam theory edge stress")
    if np.isfinite(d_fem):
        ax.axvline(d_fem, color="C3", ls=":", label=f"FEM: d = {d_fem:.2f} mm")
    ax.axvline(d_th, color="k", ls=":", label=f"Theory: d = {d_th:.2f} mm")
    ax.set_xlabel("Hole diameter d [mm]")
    ax.set_ylabel("$\\sigma_x$ [MPa]")
    ax.set_title(f"Bending (M = {load.magnitude:g} N mm): notch vs edge stress, mesh {nox}x{noy}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "diameter_search_bending.png", dpi=150)
    plt.close(fig)
    _write_csv(folder / "diameter_search_bending.csv",
               ["d", "notch_fem", "edge_fem", "notch_theory", "edge_theory"],
               np.column_stack([d_values, notch, edge, notch_th, edge_th]).tolist())
    return d_fem, d_th


def run_all(folder="study", plate: PlateData | None = None, quick=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plate = plate or PlateData()
    lines = [f"Plate L x H = {plate.L:g} x {plate.H:g} mm, t = {plate.t} mm, d = {plate.d} mm"]
    # quick=True uses small meshes only (tests / CI)
    series = {k: v[:4] for k, v in SERIES.items()} if quick else SERIES
    for kind in ("tension", "bending"):
        rows = convergence(kind, plate, folder, series=series)
        best = rows[len(series["both x and y"]) - 1]
        lines.append(f"{kind:8s}: finest mesh {best[1]}x{best[2]} ({best[3]} el.)  "
                     f"Kt element {best[6]:.4f}, nodal {best[7]:.4f}, theory {best[8]:.4f}")
        line_distribution(kind, plate, folder,
                          meshes=((8, 6), (14, 10)) if quick else ((8, 6), (14, 10), (20, 15), (32, 24), (50, 38)))
    d_fem, d_th = diameter_search(plate, folder, nox=14 if quick else 40, noy=10 if quick else 30,
                                  d_values=np.arange(40, 62.5, 5.0) if quick else None)
    lines.append(f"Bending: notch stress = edge stress at d = {d_fem:.2f} mm (FEM), "
                 f"{d_th:.2f} mm (theory, Kt = 2)")
    summary = "\n".join(lines)
    (folder / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary
