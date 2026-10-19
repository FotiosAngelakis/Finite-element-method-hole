"""Non-interactive pipeline: preprocessor -> solver -> postprocessor.

Each stage communicates with the next only through the text files in the
output folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import io, plots
from .data import PlateData
from .loads import LoadCase, apply_load_case
from .mesh import Mesh, generate_mesh
from .postprocess import NodalResults, NotchResult, notch_factor, recover_nodal
from .solver import Results, solve

# Result menus
ELEMENT_RESULTS = [
    "Displacement on x", "Displacement on y", "Total Displacement",
    "ex", "ey", "gxy", "sx", "sy", "txy", "vonMises", "s1", "s2",
]
NODAL_RESULTS = ["ex", "ey", "gxy", "sx", "sy", "txy", "vonMises", "distribution of stress x"]

_LABELS = {
    "ex": ("Strain $\\varepsilon_x$", "[-]"),
    "ey": ("Strain $\\varepsilon_y$", "[-]"),
    "gxy": ("Strain $\\gamma_{xy}$", "[-]"),
    "sx": ("Stress $\\sigma_x$", "[MPa]"),
    "sy": ("Stress $\\sigma_y$", "[MPa]"),
    "txy": ("Stress $\\tau_{xy}$", "[MPa]"),
    "vonMises": ("Von Mises stress", "[MPa]"),
}
_COMP = {"ex": ("strains", 0), "ey": ("strains", 1), "gxy": ("strains", 2),
         "sx": ("stresses", 0), "sy": ("stresses", 1), "txy": ("stresses", 2)}


# ------------------------------------------------------------------ stages
def preprocess(
    folder, prefix: str, plate: PlateData, load: LoadCase,
    nox=20, noy=15, ratio_x=0.9, ratio_y=1.0, F=None, BC=None,
):
    """Mesh, loads, supports -> text files. Returns (mesh, F, BC)."""
    mesh = generate_mesh(plate.a, plate.b, plate.d, nox, noy, ratio_x, ratio_y)
    F0, BC0 = apply_load_case(mesh, plate, load)
    F = F0 if F is None else F
    BC = BC0 if BC is None else BC
    io.write_preprocessor(folder, prefix, mesh, plate, load, F, BC)
    return mesh, F, BC


def run_solver(folder, prefix: str, method="elimination", penalty=1e7):
    """Read preprocessor files, solve, write result files."""
    mesh, plate, load, F, BC = io.read_model(folder, prefix)
    res = solve(mesh, F, BC, plate.E, plate.v, plate.t, method=method, penalty=penalty)
    io.write_solver(folder, prefix, mesh, plate, F, BC, res)
    return mesh, plate, load, F, BC, res


@dataclass
class PostData:
    mesh: Mesh
    plate: PlateData
    load: LoadCase
    F: np.ndarray
    BC: np.ndarray
    res: Results
    nodal: NodalResults
    notch: NotchResult


def load_post(folder, prefix: str) -> PostData:
    mesh, plate, load, F, BC = io.read_model(folder, prefix)
    res = io.read_results(folder, prefix, mesh)
    nodal = recover_nodal(mesh, plate, res)
    notch = notch_factor(mesh, plate, load, res, nodal)
    return PostData(mesh, plate, load, F, BC, res, nodal, notch)


# ----------------------------------------------------------------- figures
def element_figure(pd: PostData, key: str, scale=None):
    m, r = pd.mesh, pd.res
    sc = plots.default_scale(m, r) if scale is None else scale
    if key in ("Displacement on x", "Displacement on y", "Total Displacement"):
        vals = {"Displacement on x": r.U[:, 0], "Displacement on y": r.U[:, 1],
                "Total Displacement": r.U_total}[key]
        return plots.plot_nodal_field(m, vals, f"{key} (deformed x{sc:.0f})", r, sc, "[mm]")
    if key == "s1":
        return plots.plot_principal(m, r, 1)
    if key == "s2":
        return plots.plot_principal(m, r, 2)
    title, unit = _LABELS[key]
    vals = r.von_mises if key == "vonMises" else getattr(r, _COMP[key][0])[:, _COMP[key][1]]
    return plots.plot_element_field(m, vals, f"{title} - element values", r, sc, unit)


def nodal_figure(pd: PostData, key: str, scale=None):
    m, r, n = pd.mesh, pd.res, pd.nodal
    sc = plots.default_scale(m, r) if scale is None else scale
    if key == "distribution of stress x":
        return plots.plot_line_arrows(m, n.line)
    title, unit = _LABELS[key]
    vals = n.von_mises if key == "vonMises" else getattr(n, _COMP[key][0])[:, _COMP[key][1]]
    return plots.plot_nodal_field(m, vals, f"{title} - nodal (recovered)", r, sc, unit)


def save(fig, folder, name: str) -> Path:
    out = Path(folder) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fname = out / (name.replace(" ", "_").replace("/", "-") + ".png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fname


def postprocess_all(folder, prefix: str, show=False) -> PostData:
    """Produce every figure of the postprocessor and save them as PNG."""
    pd = load_post(folder, prefix)
    makers = {"00_model": lambda: plots.plot_model(pd.mesh, pd.F, pd.BC),
              "01_deformed": lambda: plots.plot_deformed(pd.mesh, pd.res)}
    for i, key in enumerate(ELEMENT_RESULTS):
        makers[f"{10 + i:02d}_element_{key}"] = lambda key=key: element_figure(pd, key)
    for i, key in enumerate(NODAL_RESULTS):
        makers[f"{30 + i:02d}_nodal_{key}"] = lambda key=key: nodal_figure(pd, key)
    makers["40_sx_symmetry_line"] = lambda: plots.plot_line_distribution(
        [("FEM (recovered)", pd.nodal.line.y, pd.nodal.line.sx)],
        theory=pd.notch.s_max_theory if pd.load.kind == "tension" else None,
    )
    for name, make in makers.items():
        fig = make()
        save(fig, folder, name)
        if not show:
            plt.close(fig)
    if show:
        plt.show()
        plt.close("all")
    report = Path(folder) / f"{prefix}_NotchFactor.txt"
    report.write_text(f"Load case: {pd.load.kind}, magnitude {pd.load.magnitude}\n"
                      f"d = {pd.plate.d} mm, H = {pd.plate.H} mm, d/H = {pd.plate.d / pd.plate.H:.4f}\n"
                      + pd.notch.summary() + "\n", encoding="utf-8")
    return pd
