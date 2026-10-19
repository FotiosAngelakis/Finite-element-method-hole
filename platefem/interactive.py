"""Interactive wizard: Preprocessor -> Solver -> Postprocessor.

Inputs are asked with terminal prompts (press Enter to accept the default
shown in brackets); supports and loads can be defined by clicking nodes on a
matplotlib figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import io, pipeline, plots
from .data import DEFAULTS, PlateData
from .loads import LoadCase, apply_load_case
from .mesh import generate_mesh
from .postprocess import edge_stress_theory


# ---------------------------------------------------------------- prompts
def ask(prompt: str, default, cast=float, check=None, err="Wrong input!"):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            raw = str(default)
        try:
            val = cast(raw)
            if check is None or check(val):
                return val
        except ValueError:
            pass
        print(f"  {err}")


def menu(prompt: str, options: list[str], default: int = 1) -> int:
    """Numbered list selection (1-based). Returns the chosen index (1-based)."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i:2d}) {opt}")
    return ask("Selection", default, int, lambda k: 1 <= k <= len(options),
               "A selection is needed!")


def yes(prompt: str, default=True) -> bool:
    return menu(prompt, ["Yes", "No"], 1 if default else 2) == 1


def gui_available() -> bool:
    return matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template")


def show(fig, folder, name):
    """Show a figure and wait until it is closed."""
    fname = pipeline.save(fig, folder, name)
    if gui_available():
        print("  (close the figure window to continue)")
        plt.show(block=True)
    else:
        print(f"  Figure saved to {fname}")
    plt.close(fig)


# ----------------------------------------------------------- node picking
def _pick_with_mouse(mesh, F, BC, what: str):
    picked = []
    fig = plots.plot_model(mesh, F, BC, title=f"Click nodes to define {what} - "
                           "right click / Enter when finished")
    ax = fig.axes[0]
    mode = menu("Picking mode", ["Single nodes (one click each)",
                                 "All nodes inside a box (two clicks: opposite corners)"])
    plt.show(block=False)
    while True:
        pts = fig.ginput(1 if mode == 1 else 2, timeout=0, mouse_stop=3)
        if not pts or (mode == 2 and len(pts) < 2):
            break
        if mode == 1:
            nodes = [mesh.nearest_node(*pts[0])]
        else:
            (x1, y1), (x2, y2) = pts
            c = mesh.coords
            nodes = list(np.flatnonzero((c[:, 0] >= min(x1, x2)) & (c[:, 0] <= max(x1, x2))
                                        & (c[:, 1] >= min(y1, y2)) & (c[:, 1] <= max(y1, y2))))
        if not nodes:
            continue
        ax.scatter(*mesh.coords[nodes].T, s=60, facecolors="none", edgecolors="k", zorder=6)
        fig.canvas.draw_idle()
        print(f"  Selected node(s): {', '.join(str(n + 1) for n in nodes)}")
        picked.append(nodes)
        if not yes("Pick more nodes?"):
            break
    plt.close(fig)
    return picked


def _pick_by_id(mesh):
    picked = []
    print("  No GUI available - enter node ids (1-based, comma separated), empty to stop.")
    print("  Tip: node ids are listed in the Nodes file / model figure.")
    while True:
        raw = input("  Node ids: ").strip()
        if not raw:
            break
        try:
            ids = [int(s) - 1 for s in raw.replace(";", ",").split(",") if s.strip()]
            if all(0 <= i < mesh.n_nodes for i in ids):
                picked.append(ids)
                continue
        except ValueError:
            pass
        print("  Wrong input!")
    return picked


def define_supports(mesh, F, BC):
    BC = BC.copy()
    if yes("Remove the default supports first?"):
        BC[:] = 0
    groups = _pick_with_mouse(mesh, F, BC, "supports") if gui_available() else _pick_by_id(mesh)
    for nodes in groups:
        print(f"  Supports for node(s) {', '.join(str(n + 1) for n in nodes)} (1 = fixed, 0 = free)")
        bx = ask("    BC at x direction", 0, int, lambda k: k in (0, 1))
        by = ask("    BC at y direction", 0, int, lambda k: k in (0, 1))
        BC[nodes] = (bx, by)
    return BC


def define_loads(mesh, F, BC):
    F = F.copy()
    if yes("Remove the default loads first?", default=False):
        F[:] = 0
    groups = _pick_with_mouse(mesh, F, BC, "loads") if gui_available() else _pick_by_id(mesh)
    for nodes in groups:
        print(f"  Force on EACH of node(s) {', '.join(str(n + 1) for n in nodes)} [N]")
        fx = ask("    Fx", 0.0)
        fy = ask("    Fy", 0.0)
        F[nodes] += (fx, fy)
    return F


# ------------------------------------------------------------------ stages
def preprocessor(folder: Path, prefix: str):
    print("\n--------------------------- PREPROCESSOR ---------------------------")
    D = DEFAULTS
    pos = lambda x: x > 0  # noqa: E731
    while True:
        print("Plate geometry (the hole is at the centre of the plate)")
        L = ask("  Plate length in x, L [mm]", D["length"], float, pos)
        H = ask("  Plate width in y, H [mm]", D["width"], float, pos)
        t = ask("  Plate thickness, t [mm]", D["t"], float, pos)
        d = ask("  Hole diameter, d [mm]", D["d"], float, lambda x: 0 < x < min(L, H))
        print("Material")
        E = ask("  Young's modulus, E [MPa]", D["E"], float, pos)
        v = ask("  Poisson ratio, v [-]", D["v"], float, lambda x: 0 <= x < 0.5)
        try:
            plate = PlateData.from_dimensions(L, H, t, d, E, v)
            break
        except ValueError as exc:
            print(f"  {exc}")
    print(f"  Shear modulus G = E / (2(1+v)) = {plate.G:.1f} MPa")

    while True:
        nox = ask("Number of nodes in x direction", 20, int, lambda k: k >= 5)
        noy = ask("Number of nodes in y direction", 15, int, lambda k: k >= 2)
        rx = ask("Ratio in x direction (<1 refines towards the hole)", 0.9, float, lambda r: r > 0)
        ry = ask("Ratio in y direction", 1.0, float, lambda r: r > 0)
        try:
            mesh = generate_mesh(plate.a, plate.b, plate.d, nox, noy, rx, ry)
            break
        except ValueError as exc:
            print(f"  {exc}")
    print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} CST elements")

    kind = menu("Type of loading", ["Tension", "Bending"])
    if kind == 1:
        P = ask("Magnitude of tensile force P [N]", 2000.0)
        sup = menu("Supports", ["Symmetric: load both ends, restrain symmetry lines (recommended)",
                                "Load at x = 0, clamp the edge x = L"])
        load = LoadCase("tension", P, "symmetric" if sup == 1 else "clamped")
    else:
        M = ask("Magnitude of bending moment M [N mm]", 10000.0)
        load = LoadCase("bending", M)

    F, BC = apply_load_case(mesh, plate, load)
    fig = plots.plot_model(mesh, F, BC)
    show(fig, folder, "00_model")

    if menu("Boundary conditions", ["Default boundary conditions",
                                    "Define boundary conditions (mouse)"]) == 2:
        BC = define_supports(mesh, F, BC)
    if menu("Loads", ["Default loads", "Define / add nodal loads (mouse)"]) == 2:
        F = define_loads(mesh, F, BC)
    if menu("Boundary conditions / loads changed?", ["Show the final model", "Skip"], 2) == 1:
        show(plots.plot_model(mesh, F, BC), folder, "00_model_final")

    io.write_preprocessor(folder, prefix, mesh, plate, load, F, BC)
    print(f"  Files written to {Path(folder).resolve()}")


def solver(folder: Path, prefix: str):
    print("\n------------------------------ SOLVER ------------------------------")
    try:
        mesh, plate, load, F, BC, res = pipeline.run_solver(folder, prefix)
    except RuntimeError as exc:
        print(f"  ERROR: {exc}")
        return False
    print(f"  Solved {2 * mesh.n_nodes} DOFs. max |u| = {res.U_total.max():.5e} mm, "
          f"max von Mises = {res.von_mises.max():.4f} MPa")
    return True


def postprocessor(folder: Path, prefix: str):
    print("\n--------------------------- POSTPROCESSOR --------------------------")
    pd = pipeline.load_post(folder, prefix)
    show(plots.plot_deformed(pd.mesh, pd.res), folder, "01_deformed")

    opts = pipeline.ELEMENT_RESULTS + ["none"]
    while True:
        k = menu("Select result to plot", opts, len(opts))
        if k < len(opts):
            key = opts[k - 1]
            show(pipeline.element_figure(pd, key), folder, f"element_{key}")
        if not yes("Do you want to plot another result?", default=False):
            break

    n = pd.notch
    kind = "Tension" if pd.load.kind == "tension" else "Bending"
    print(f"\n  Notch factor for {kind} (element values): Kt = {n.kt_element:.4f}"
          f"   (theory {n.kt_theory:.4f})")

    if not yes("Do you want to interpolate (recover nodal values) for more accurate results "
               "and notch factor?"):
        return
    opts = pipeline.NODAL_RESULTS + ["none"]
    while True:
        k = menu("Select result to plot", opts, len(opts))
        if k < len(opts):
            key = opts[k - 1]
            show(pipeline.nodal_figure(pd, key), folder, f"nodal_{key}")
        if not yes("Do you want to plot another result?", default=False):
            break
    print("\n" + n.summary())
    if pd.load.kind == "bending":
        notch, edge = n.s_max_nodal, n.edge_stress
        print(f"  Notch stress {notch:.4f} MPa vs edge stress {edge:.4f} MPa "
              f"(beam theory edge {edge_stress_theory(pd.plate, pd.load):.4f} MPa)")
        if abs(notch - edge) / max(edge, 1e-12) < 0.01:
            print("  You found the d!  (notch stress = edge stress)")
    (Path(folder) / f"{prefix}_NotchFactor.txt").write_text(n.summary() + "\n", encoding="utf-8")


def run(folder="results", prefix="Plate"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    print("FEM - 2D plate with a central hole (plane stress, CST elements)")
    print("Units: N, mm, MPa.  Press Enter to accept the [default] values.")
    preprocessor(folder, prefix)
    if menu("Next step", ["Continue to Solver", "Terminate process"]) != 1:
        return
    if not solver(folder, prefix):
        return
    if menu("Next step", ["Continue to PostProcessor", "Terminate process"]) != 1:
        return
    postprocessor(folder, prefix)
    print("\nDone.")
