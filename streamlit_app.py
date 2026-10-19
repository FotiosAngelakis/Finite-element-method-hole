"""Streamlit web app for platefem.

Run locally:   streamlit run streamlit_app.py
Deploy:        share.streamlit.io -> select this repository, main file streamlit_app.py
"""

from __future__ import annotations

import io as _io
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from platefem import io as pio  # noqa: E402
from platefem import pipeline, plots, study  # noqa: E402
from platefem.data import DEFAULTS, PlateData  # noqa: E402
from platefem.loads import LoadCase, apply_load_case  # noqa: E402
from platefem.mesh import generate_mesh  # noqa: E402
from platefem.postprocess import notch_factor, recover_nodal  # noqa: E402
from platefem.solver import solve  # noqa: E402

st.set_page_config(page_title="Plate with a hole - FEM", page_icon="🔩", layout="wide")


# ---------------------------------------------------------------- caching
@st.cache_data(show_spinner=False)
def build_mesh(a, b, d, nox, noy, rx, ry):
    return generate_mesh(a, b, d, nox, noy, rx, ry)


@st.cache_data(show_spinner=False)
def run_analysis(plate_args, mesh_args, load_args, F_bytes, BC_bytes, method):
    plate = PlateData(**plate_args)
    mesh = build_mesh(plate.a, plate.b, plate.d, *mesh_args)
    load = LoadCase(**load_args)
    F = np.frombuffer(F_bytes).reshape(-1, 2)
    BC = np.frombuffer(BC_bytes, dtype=int).reshape(-1, 2)
    res = solve(mesh, F, BC, plate.E, plate.v, plate.t, method=method)
    nodal = recover_nodal(mesh, plate, res)
    notch = notch_factor(mesh, plate, load, res, nodal)
    return pipeline.PostData(mesh, plate, load, F, BC, res, nodal, notch)


@st.cache_data(show_spinner=False)
def run_study(plate_args, quick):
    folder = Path(tempfile.mkdtemp(prefix="platefem_study_"))
    summary = study.run_all(folder, PlateData(**plate_args), quick=quick)
    files = {p.name: p.read_bytes() for p in sorted(folder.iterdir())}
    return summary, files


def show_fig(fig):
    st.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Input data")
D = DEFAULTS
with st.sidebar.expander("Plate geometry", expanded=True):
    L = st.number_input("Plate length in x, L [mm]", 1.0, 10000.0, D["length"], 5.0)
    H = st.number_input("Plate width in y, H [mm]", 1.0, 10000.0, D["width"], 5.0)
    t = st.number_input("Thickness, t [mm]", 0.01, 1000.0, D["t"], 0.5)
    d = st.number_input("Hole diameter, d [mm]", 0.1, 10000.0, D["d"], 0.5,
                        help="Circular hole at the centre of the plate")
with st.sidebar.expander("Material", expanded=True):
    E = st.number_input("Young's modulus, E [MPa]", 1.0, 1e7, D["E"], 1000.0)
    v = st.number_input("Poisson ratio, ν [-]", 0.0, 0.499, D["v"], 0.01)
with st.sidebar.expander("Mesh", expanded=True):
    st.caption("Nodes per side of one quarter of the plate. A ratio < 1 makes every "
               "segment that much shorter than the previous one (refines towards the hole).")
    nox = st.slider("Nodes in x direction", 5, 60, 20)
    noy = st.slider("Nodes in y direction", 2, 50, 15)
    rx = st.number_input("Spacing ratio in x", 0.5, 1.5, 0.9, 0.01)
    ry = st.number_input("Spacing ratio in y", 0.5, 1.5, 1.0, 0.01)
with st.sidebar.expander("Loading", expanded=True):
    kind = st.radio("Load case", ["Tension", "Bending"], horizontal=True).lower()
    if kind == "tension":
        mag = st.number_input("Tensile force, P [N]", value=2000.0, step=100.0)
        sup = st.radio("Supports", ["Symmetric (both ends loaded)", "Clamped right edge"],
                       help="Symmetric: P on both ends, u_x = 0 on x = L/2 and u_y = 0 on y = H/2. "
                            "Clamped: P at x = 0, u_x = 0 on x = L, u_y = 0 at its middle.")
        supports = "symmetric" if sup.startswith("Symmetric") else "clamped"
    else:
        mag = st.number_input("Bending moment, M [N·mm]", value=10000.0, step=1000.0)
        supports = "symmetric"
with st.sidebar.expander("Solver"):
    method = st.radio("Supports applied by", ["elimination", "penalty"],
                      help="Exact elimination of the fixed DOFs, or a large penalty "
                           "(1e7) added to their diagonal stiffness terms.")
st.sidebar.caption("Units: N, mm, MPa")

# ---------------------------------------------------------------- model
try:
    plate = PlateData.from_dimensions(L, H, t, d, E, v)
    mesh = build_mesh(plate.a, plate.b, plate.d, nox, noy, rx, ry)
except ValueError as exc:
    st.error(f"Invalid input: {exc}")
    st.stop()
load = LoadCase(kind, mag, supports)
F0, BC0 = apply_load_case(mesh, plate, load)

# user modifications of supports / loads are kept per model definition
model_key = (L, H, t, d, nox, noy, rx, ry, kind, mag, supports)
if st.session_state.get("model_key") != model_key:
    st.session_state.model_key = model_key
    st.session_state.F = F0.copy()
    st.session_state.BC = BC0.copy()
F, BC = st.session_state.F, st.session_state.BC

st.title("Plate with a central hole: plane-stress FEM")
st.caption(f"{mesh.n_nodes} nodes · {mesh.n_elements} constant-strain triangles · "
           f"d/H = {plate.d / plate.H:.3f} · G = {plate.G:.0f} MPa")

tabs = st.tabs(["Model", "Results", "Notch factor", "Mesh study", "Downloads"])


# ================================================================ MODEL
def model_figure(mesh, F, BC):
    c, e = mesh.coords, mesh.elements
    seg = c[e][:, [0, 1, 2, 0], :]
    xs = np.concatenate([seg[:, :, 0], np.full((len(e), 1), np.nan)], 1).ravel()
    ys = np.concatenate([seg[:, :, 1], np.full((len(e), 1), np.nan)], 1).ravel()
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="#7aa6d6", width=0.6),
                               hoverinfo="skip", name="Elements"))
    kind_of = np.select([(BC[:, 0] == 1) & (BC[:, 1] == 1), BC[:, 0] == 1, BC[:, 1] == 1],
                        ["Fixed x,y", "Fixed x", "Fixed y"], "Free")
    styles = {"Free": ("#d62728", 5, "circle"), "Fixed x": ("#2ca02c", 9, "triangle-right"),
              "Fixed y": ("#ff7f0e", 9, "triangle-up"), "Fixed x,y": ("#e377c2", 10, "square")}
    for name, (col, size, sym) in styles.items():
        idx = np.flatnonzero(kind_of == name)
        if len(idx) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=c[idx, 0], y=c[idx, 1], mode="markers", name=name,
            marker=dict(color=col, size=size, symbol=sym), customdata=idx + 1,
            text=[f"Fx={F[i, 0]:.3g}, Fy={F[i, 1]:.3g}" for i in idx],
            hovertemplate="node %{customdata}<br>(%{x:.3f}, %{y:.3f})<br>%{text}<extra></extra>"))
    loaded = np.flatnonzero(np.abs(F).sum(1) > 0)
    if len(loaded):
        sc = 0.1 * np.ptp(c[:, 0]) / np.abs(F[loaded]).max()
        ax_ = np.column_stack([c[loaded, 0], c[loaded, 0] + F[loaded, 0] * sc,
                               np.full(len(loaded), np.nan)]).ravel()
        ay_ = np.column_stack([c[loaded, 1], c[loaded, 1] + F[loaded, 1] * sc,
                               np.full(len(loaded), np.nan)]).ravel()
        fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color="black", width=2),
                                 name="Forces", hoverinfo="skip"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), dragmode="select",
                      legend=dict(orientation="h", y=-0.08),
                      xaxis_title="x [mm]", yaxis_title="y [mm]")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


with tabs[0]:
    st.subheader("Mesh, supports and loads")
    st.caption("Select nodes with the box or lasso tool (or click a node), then apply "
               "supports or loads below. Hover a node to see its id and load.")
    event = st.plotly_chart(model_figure(mesh, F, BC), width="stretch",
                            on_select="rerun", selection_mode=("points", "box", "lasso"),
                            key=f"model_{hash(model_key)}")
    selected = []
    try:
        selected = sorted({int(p["customdata"]) - 1 for p in event.selection.points
                           if p.get("customdata") is not None})
    except (AttributeError, KeyError, TypeError):
        selected = []
    typed = st.text_input("…or type node ids (comma separated)",
                          placeholder="e.g. 1, 21, 41")
    if typed.strip():
        try:
            ids = [int(s) - 1 for s in typed.replace(";", ",").split(",") if s.strip()]
            selected = sorted(set(selected) | {i for i in ids if 0 <= i < mesh.n_nodes})
        except ValueError:
            st.warning("Node ids must be integers")
    st.write(f"**Selected nodes ({len(selected)}):** "
             + (", ".join(str(i + 1) for i in selected[:60]) + (" …" if len(selected) > 60 else "")
                if selected else "none"))

    c1, c2 = st.columns(2)
    with c1, st.form("bc_form"):
        st.markdown("**Supports for the selected nodes**")
        fx_ = st.checkbox("Fix x displacement (u_x = 0)")
        fy_ = st.checkbox("Fix y displacement (u_y = 0)")
        if st.form_submit_button("Apply supports", disabled=not selected):
            BC[selected] = (int(fx_), int(fy_))
            st.rerun()
    with c2, st.form("load_form"):
        st.markdown("**Nodal force on each selected node**")
        cc1, cc2 = st.columns(2)
        Fx_ = cc1.number_input("Fx [N]", value=0.0)
        Fy_ = cc2.number_input("Fy [N]", value=0.0)
        mode = st.radio("Mode", ["Add to existing", "Replace"], horizontal=True)
        if st.form_submit_button("Apply force", disabled=not selected):
            if mode == "Replace":
                F[selected] = (Fx_, Fy_)
            else:
                F[selected] += (Fx_, Fy_)
            st.rerun()
    b1, b2, b3 = st.columns(3)
    if b1.button("Reset to default supports and loads"):
        st.session_state.F, st.session_state.BC = F0.copy(), BC0.copy()
        st.rerun()
    if b2.button("Remove all supports"):
        BC[:] = 0
        st.rerun()
    if b3.button("Remove all loads"):
        F[:] = 0
        st.rerun()
    st.caption(f"Resultant applied force: Fx = {F[:, 0].sum():.4g} N, Fy = {F[:, 1].sum():.4g} N · "
               f"fixed DOFs: {int(BC.sum())}")

# ---------------------------------------------------------------- solve
try:
    with st.spinner("Solving…"):
        pd = run_analysis(dict(t=plate.t, d=plate.d, E=plate.E, v=plate.v, a=plate.a, b=plate.b),
                          (nox, noy, rx, ry), dict(kind=kind, magnitude=mag, supports=supports),
                          np.ascontiguousarray(F, dtype=float).tobytes(),
                          np.ascontiguousarray(BC, dtype=int).tobytes(), method)
    solved = True
except RuntimeError as exc:
    solved = False
    for tab in tabs[1:3] + tabs[4:]:
        tab.error(f"Cannot solve: {exc}")

# ================================================================ RESULTS
if solved:
    res = pd.res
    with tabs[1]:
        m1, m2, m3 = st.columns(3)
        m1.metric("max |u|", f"{res.U_total.max():.4e} mm")
        m2.metric("max von Mises", f"{res.von_mises.max():.4f} MPa")
        m3.metric("max σ1", f"{res.principal[:, 0].max():.4f} MPa")
        c1, c2, c3 = st.columns([2, 2, 1])
        group = c1.radio("Values", ["Element values (one colour per element)",
                                    "Nodal values (continuous, recovered)"])
        if group.startswith("Element"):
            options = ["Deformed shape"] + pipeline.ELEMENT_RESULTS
        else:
            options = pipeline.NODAL_RESULTS + ["σx on the symmetry axis (plot)"]
        key = c2.selectbox("Result", options, index=options.index("sx"))
        auto = plots.default_scale(pd.mesh, res)
        scale = c3.number_input("Deformation scale", 0.0, 1e9, float(round(auto)), help=(
            "Displacements are multiplied by this factor for plotting (0 = undeformed)"))
        if key == "Deformed shape":
            fig = plots.plot_deformed(pd.mesh, res, scale=scale)
        elif key == "σx on the symmetry axis (plot)":
            fig = plots.plot_line_distribution(
                [("FEM (recovered)", pd.nodal.line.y, pd.nodal.line.sx)],
                theory=pd.notch.s_max_theory if kind == "tension" else None)
        elif group.startswith("Element"):
            fig = pipeline.element_figure(pd, key, scale=scale)
        else:
            fig = pipeline.nodal_figure(pd, key, scale=scale)
        show_fig(fig)

    # ============================================================ NOTCH
    with tabs[2]:
        n = pd.notch
        st.subheader(f"Notch factor - {kind}")
        if kind == "tension":
            st.latex(r"\sigma_{nom}=\frac{P}{t\,(H-d)}\qquad K_t=\frac{\sigma_{max}}{\sigma_{nom}}"
                     r"\qquad K_{t,theory}=3-3.14\tfrac{d}{H}+3.667\left(\tfrac{d}{H}\right)^2"
                     r"-1.527\left(\tfrac{d}{H}\right)^3")
        else:
            st.latex(r"\sigma_{nom}=\frac{6\,M\,d}{(H^3-d^3)\,t}\qquad K_t="
                     r"\frac{\sigma_{max}}{\sigma_{nom}}\qquad K_{t,theory}\approx 2")
        c = st.columns(4)
        c[0].metric("σ nominal", f"{n.s_nominal:.4f} MPa")
        c[1].metric("Kt (element values)", f"{n.kt_element:.4f}",
                    f"{100 * (n.kt_element / n.kt_theory - 1):+.1f}% vs theory", delta_color="off")
        c[2].metric("Kt (recovered nodal)", f"{n.kt_nodal:.4f}",
                    f"{100 * (n.kt_nodal / n.kt_theory - 1):+.1f}% vs theory", delta_color="off")
        c[3].metric("Kt theory (Peterson)", f"{n.kt_theory:.4f}")
        st.code(n.summary(), language=None)
        if kind == "bending":
            diff = (n.s_max_nodal - n.edge_stress) / n.edge_stress
            st.info(f"Notch stress {n.s_max_nodal:.4f} MPa vs edge stress {n.edge_stress:.4f} MPa "
                    f"({100 * diff:+.1f}%). Change d to make them equal, or run the diameter "
                    "search in the *Mesh study* tab.")
        show_fig(plots.plot_line_distribution(
            [("FEM (recovered)", pd.nodal.line.y, pd.nodal.line.sx)],
            theory=n.s_max_theory if kind == "tension" else None))

# ================================================================ STUDY
with tabs[3]:
    st.subheader("Mesh sensitivity study")
    st.markdown(
        "* **Convergence** of the notch factor and peak stress with the number of elements, "
        "refining both directions together, x only and y only (tension and bending).\n"
        "* **σx distribution on the vertical symmetry axis** for several mesh densities.\n"
        "* **Bending: hole-diameter search**: d for which the notch stress equals the edge "
        "stress, compared with theory (Kt·d = H).\n\n"
        "Uses the plate geometry and material from the sidebar; the mesh sizes are set by the study.")
    quick = st.toggle("Quick run (small meshes only)", value=False)
    if st.button("Run study", type="primary"):
        st.session_state.study_args = (
            dict(t=plate.t, d=plate.d, E=plate.E, v=plate.v, a=plate.a, b=plate.b), quick)
    if "study_args" in st.session_state:
        with st.spinner("Running the mesh study… (about 10 s)"):
            summary, files = run_study(*st.session_state.study_args)
        st.code(summary, language=None)
        for name in ("convergence_tension.png", "convergence_bending.png"):
            st.image(files[name], width="stretch")
        cA, cB = st.columns(2)
        cA.image(files["line_distribution_tension.png"], width="stretch")
        cB.image(files["line_distribution_bending.png"], width="stretch")
        st.image(files["diameter_search_bending.png"], width="stretch")
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(name, data)
        st.download_button("Download study (PNG + CSV)", buf.getvalue(), "mesh_study.zip",
                           "application/zip")

# ================================================================ DOWNLOADS
if solved:
    with tabs[4]:
        st.subheader("Result files")
        st.markdown("Text files (comma separated, one header line): nodes, elements, data, "
                    "displacements, reactions, strains, stresses (σx, σy, τxy, σ1, σ2, φ, von Mises) "
                    "and the notch-factor summary. They can be read back with "
                    "`platefem postprocess -o <folder>`.")
        with tempfile.TemporaryDirectory() as tmp:
            pio.write_preprocessor(tmp, "Plate", pd.mesh, pd.plate, pd.load, pd.F, pd.BC)
            pio.write_solver(tmp, "Plate", pd.mesh, pd.plate, pd.F, pd.BC, pd.res)
            # keep the full data file written by the preprocessor
            Path(tmp, "Plate_NotchFactor.txt").write_text(pd.notch.summary() + "\n")
            buf = _io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(Path(tmp).iterdir()):
                    z.write(p, p.name)
        st.download_button("Download all result files (.zip)", buf.getvalue(),
                           "plate_results.zip", "application/zip", type="primary")
