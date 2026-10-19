"""Command line interface.

    platefem                      interactive wizard (step-by-step prompts)
    platefem run   [options]      preprocessor + solver + postprocessor, no prompts
    platefem preprocess / solve / postprocess [options]   single stages
    platefem study [options]      mesh sensitivity studies
"""

from __future__ import annotations

import argparse
import sys

from .data import DEFAULTS, PlateData
from .loads import LoadCase


def _common(p):
    p.add_argument("-o", "--out", default="results", help="output folder (default: results)")
    p.add_argument("--prefix", default="Plate", help="file name prefix (default: Plate)")


def _plate_args(p):
    D = DEFAULTS
    g = p.add_argument_group("plate geometry and material")
    g.add_argument("-L", "--length", type=float, default=D["length"],
                   help=f"total plate length in x [mm] (default {D['length']:g})")
    g.add_argument("-H", "--width", type=float, default=D["width"],
                   help=f"total plate width in y [mm] (default {D['width']:g})")
    g.add_argument("-t", "--thickness", type=float, default=D["t"],
                   help=f"plate thickness [mm] (default {D['t']:g})")
    g.add_argument("-d", "--diameter", type=float, default=D["d"],
                   help=f"diameter of the central hole [mm] (default {D['d']:g})")
    g.add_argument("-E", "--youngs-modulus", dest="E", type=float, default=D["E"],
                   help=f"Young's modulus [MPa] (default {D['E']:g})")
    g.add_argument("--nu", "--poisson", dest="nu", type=float, default=D["v"],
                   help=f"Poisson ratio [-] (default {D['v']:g})")


def _model_args(p):
    _plate_args(p)
    p.add_argument("--nox", type=int, default=20, help="nodes in x per quarter (default 20)")
    p.add_argument("--noy", type=int, default=15, help="nodes in y per quarter (default 15)")
    p.add_argument("--rx", type=float, default=0.9, help="spacing ratio in x (default 0.9)")
    p.add_argument("--ry", type=float, default=1.0, help="spacing ratio in y (default 1.0)")
    p.add_argument("--load", choices=["tension", "bending"], default="tension")
    p.add_argument("--magnitude", type=float,
                   help="tensile force P [N] (default 2000) or moment M [N mm] (default 10000)")
    p.add_argument("--supports", choices=["symmetric", "clamped"], default="symmetric",
                   help="tension supports: symmetric (default) or clamped right edge")


def _solver_args(p):
    p.add_argument("--method", choices=["elimination", "penalty"], default="elimination",
                   help="how supports are applied: exact elimination or a penalty on the diagonal")
    p.add_argument("--penalty", type=float, default=1e7)


def _plate(args) -> PlateData:
    return PlateData.from_dimensions(args.length, args.width, args.thickness,
                                     args.diameter, args.E, args.nu)


def _load(args) -> LoadCase:
    mag = args.magnitude
    if mag is None:
        mag = 2000.0 if args.load == "tension" else 10000.0
    return LoadCase(args.load, mag, args.supports)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="platefem", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("interactive", help="interactive wizard (default)")
    _common(p)

    p = sub.add_parser("run", help="run all three stages without prompts")
    _common(p)
    _model_args(p)
    _solver_args(p)
    p.add_argument("--show", action="store_true", help="also open the figures")

    p = sub.add_parser("preprocess", help="mesh + loads + supports -> text files")
    _common(p)
    _model_args(p)

    p = sub.add_parser("solve", help="solve the model in the output folder")
    _common(p)
    _solver_args(p)

    p = sub.add_parser("postprocess", help="figures + notch factor from the result files")
    _common(p)
    p.add_argument("--show", action="store_true")

    p = sub.add_parser("study", help="mesh study: convergence, sigma_x on symmetry axis, d-search")
    p.add_argument("-o", "--out", default="study")
    _plate_args(p)
    p.add_argument("--quick", action="store_true", help="small meshes only (fast)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.cmd or "interactive"

    if cmd == "interactive":
        from . import interactive
        try:
            interactive.run(getattr(args, "out", "results"), getattr(args, "prefix", "Plate"))
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 1
        return 0

    import matplotlib
    if not getattr(args, "show", False):
        matplotlib.use("Agg")
    from . import pipeline

    if cmd in ("run", "preprocess"):
        plate, load = _plate(args), _load(args)
        mesh, _, _ = pipeline.preprocess(args.out, args.prefix, plate, load,
                                         args.nox, args.noy, args.rx, args.ry)
        print(f"Preprocessor: {mesh.n_nodes} nodes, {mesh.n_elements} elements, d = {plate.d} mm")
    if cmd in ("run", "solve"):
        mesh, *_, res = pipeline.run_solver(args.out, args.prefix, args.method, args.penalty)
        print(f"Solver: max |u| = {res.U_total.max():.5e} mm, "
              f"max von Mises = {res.von_mises.max():.4f} MPa")
    if cmd in ("run", "postprocess"):
        pd = pipeline.postprocess_all(args.out, args.prefix, show=args.show)
        print(pd.notch.summary())
        print(f"Figures saved in {args.out}/figures")
    if cmd == "study":
        matplotlib.use("Agg")
        from . import study
        print(study.run_all(args.out, _plate(args), quick=args.quick))
        print(f"Figures and CSV tables saved in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
