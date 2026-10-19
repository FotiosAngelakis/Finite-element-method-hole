from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from platefem import (LoadCase, PlateData, apply_load_case, generate_mesh,
                      notch_factor, solve, spacing)
from platefem import io, pipeline
from platefem.cli import main
from platefem.postprocess import kt_tension_theory, nodal_average

# Reference solution: bending, d = 50 mm, 20 x 15 nodes, ratios 0.9 / 1,
# loads lumped at the loaded edges, penalty supports (1e7).
REF = Path(__file__).parent / "data" / "reference"


def _ref(name):
    return np.genfromtxt(REF / f"Reference_{name}_Plate.txt", delimiter=",", skip_header=1)


# ----------------------------------------------------------------- basics
def test_spacing_uniform_and_geometric():
    np.testing.assert_allclose(spacing(0, 10, 5, 1), [0, 2.5, 5, 7.5, 10])
    x = spacing(0, 85, 20, 0.9)
    assert x[0] == 0 and x[-1] == pytest.approx(85)
    steps = np.diff(x)
    np.testing.assert_allclose(steps[1:] / steps[:-1], 0.9)


def test_plate_data():
    p = PlateData.from_dimensions(length=170, width=100, t=4, d=42.4, E=210000, v=0.3)
    assert (p.a, p.b, p.L, p.H, p.R) == (85, 50, 170, 100, 21.2)
    assert p.G == pytest.approx(80769.23, rel=1e-6)
    assert PlateData() == p
    with pytest.raises(ValueError):
        PlateData.from_dimensions(d=120)  # hole wider than the plate
    with pytest.raises(ValueError):
        PlateData.from_dimensions(v=0.6)


def test_peterson_tension():
    assert kt_tension_theory(0.424) == pytest.approx(2.2115, abs=1e-4)


# ------------------------------------------- comparison with reference solution
def test_mesh_matches_reference():
    mesh = generate_mesh(85, 50, 50, 20, 15, 0.9, 1.0)
    N, E = _ref("Nodes"), _ref("Elements")
    assert mesh.n_nodes == len(N) == 1154
    assert mesh.n_elements == len(E) == 2128
    np.testing.assert_allclose(mesh.coords, N[:, 1:3], atol=1e-3)  # file has 6 digits
    np.testing.assert_array_equal(mesh.elements, E[:, 1:4].astype(int) - 1)
    np.testing.assert_allclose(mesh.areas, E[:, 4], rtol=1e-3)
    assert np.all(mesh.areas > 0)
    assert mesh.areas.sum() == pytest.approx(170 * 100 - np.pi * 25**2, rel=2e-3)


def test_solver_matches_reference():
    """Solve the reference model (its loads and supports) with the penalty method."""
    mesh = generate_mesh(85, 50, 50, 20, 15, 0.9, 1.0)
    N, Uref = _ref("Nodes"), _ref("Displacement")
    res = solve(mesh, N[:, 5:7], N[:, 3:5].astype(int), 210000, 0.3, 4.0,
                method="penalty", penalty=1e7)
    err = np.abs(res.U - Uref).max() / np.abs(Uref).max()
    assert err < 1e-4


def test_read_compact_files(tmp_path):
    for f in REF.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    mesh, plate, load, F, BC = io.read_model(tmp_path, "Reference")
    assert plate.d == pytest.approx(50, abs=1e-3)
    assert mesh.n_elements == 2128 and BC.sum() == 32


# ------------------------------------------------------------ mechanics
@pytest.mark.parametrize("kind", ["tension", "bending"])
def test_equilibrium_and_loads(kind):
    p = PlateData(d=42.4)
    lc = LoadCase(kind, 2000 if kind == "tension" else 10000)
    mesh = generate_mesh(p.a, p.b, p.d, 14, 10, 0.86, 1)
    F, BC = apply_load_case(mesh, p, lc)
    left = mesh.coords[:, 0] < 1e-9
    if kind == "tension":
        assert F[left, 0].sum() == pytest.approx(-2000)
    else:  # resultant moment of the left-edge forces = M, no net force
        assert F[left, 0].sum() == pytest.approx(0, abs=1e-9)
        assert -(F[left, 0] * (mesh.coords[left, 1] - p.b)).sum() == pytest.approx(-10000)
    res = solve(mesh, F, BC, p.E, p.v, p.t)
    np.testing.assert_allclose((res.reactions + F).sum(0), F.sum(0), atol=1e-8)
    assert np.abs(res.reactions.sum(0)).max() < 1e-8  # self-equilibrated loads


def test_uniform_stress_far_from_hole():
    p = PlateData(d=10)
    mesh = generate_mesh(p.a, p.b, p.d, 20, 15, 1.0, 1.0)
    F, BC = apply_load_case(mesh, p, LoadCase("tension", 2000))
    res = solve(mesh, F, BC, p.E, p.v, p.t)
    far = mesh.centroids[:, 0] < 15
    np.testing.assert_allclose(res.stresses[far, 0], 2000 / (100 * 4), rtol=0.02)


@pytest.mark.parametrize("kind,lo,hi", [("tension", 2.15, 2.5), ("bending", 1.85, 2.15)])
def test_notch_factor_reasonable(kind, lo, hi):
    p = PlateData(d=42.4)
    lc = LoadCase(kind, 2000 if kind == "tension" else 10000)
    mesh = generate_mesh(p.a, p.b, p.d, 20, 15, 0.9, 1)
    F, BC = apply_load_case(mesh, p, lc)
    res = solve(mesh, F, BC, p.E, p.v, p.t)
    n = notch_factor(mesh, p, lc, res)
    assert lo < n.kt_element < hi and lo < n.kt_nodal < hi


def test_penalty_close_to_elimination():
    p = PlateData(d=42.4)
    mesh = generate_mesh(p.a, p.b, p.d, 14, 10, 0.86, 1)
    F, BC = apply_load_case(mesh, p, LoadCase("bending", 10000))
    r1 = solve(mesh, F, BC, p.E, p.v, p.t)
    r2 = solve(mesh, F, BC, p.E, p.v, p.t, method="penalty", penalty=1e12)
    np.testing.assert_allclose(r1.U, r2.U, atol=1e-6 * np.abs(r1.U).max())


def test_singular_model_raises():
    p = PlateData(d=42.4)
    mesh = generate_mesh(p.a, p.b, p.d, 8, 5, 1, 1)
    F, BC = apply_load_case(mesh, p, LoadCase("tension", 1000))
    with pytest.raises(RuntimeError):
        solve(mesh, F, np.zeros_like(BC), p.E, p.v, p.t)


def test_nodal_average_constant():
    mesh = generate_mesh(85, 50, 42.4, 8, 5, 1, 1)
    np.testing.assert_allclose(nodal_average(mesh, np.full(mesh.n_elements, 3.0)), 3.0)


# ------------------------------------------------------------ pipeline / CLI
def test_pipeline_roundtrip(tmp_path):
    p, lc = PlateData(d=42.4), LoadCase("tension", 2000)
    mesh, F, BC = pipeline.preprocess(tmp_path, "T", p, lc, 10, 8, 0.85, 1)
    *_, res = pipeline.run_solver(tmp_path, "T")
    pd = pipeline.load_post(tmp_path, "T")
    np.testing.assert_allclose(pd.res.U, res.U, rtol=1e-5, atol=1e-12)
    assert pd.load.kind == "tension" and pd.plate.d == pytest.approx(42.4)


def test_cli_run_and_study(tmp_path):
    assert main(["run", "-o", str(tmp_path / "r"), "--nox", "10", "--noy", "8", "--rx", "0.85"]) == 0
    figs = list((tmp_path / "r" / "figures").glob("*.png"))
    assert len(figs) >= 20
    assert main(["study", "-o", str(tmp_path / "s"), "--quick"]) == 0
    assert (tmp_path / "s" / "diameter_search_bending.png").exists()


def test_interactive_defaults(tmp_path, monkeypatch):
    """Press Enter at every prompt: full preprocessor -> solver -> postprocessor run."""
    from platefem import interactive

    answers = iter([])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers, ""))
    interactive.run(tmp_path, "I")
    assert (tmp_path / "I_Stresses_Plate.txt").exists()
    assert (tmp_path / "I_NotchFactor.txt").exists()


# ------------------------------------------------------------ web app
def test_streamlit_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception and not at.error
    kt = {m.label: float(m.value) for m in at.metric if m.label.startswith("Kt")}
    assert 2.1 < kt["Kt (element values)"] < 2.5
    [r for r in at.radio if r.label == "Load case"][0].set_value("Bending").run()
    assert not at.exception and not at.error
    at.text_input[0].set_value("1, 21").run()
    [b for b in at.button if b.label == "Remove all supports"][0].click().run()
    assert any("singular" in e.value for e in at.error)
