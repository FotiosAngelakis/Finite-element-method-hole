"""Matplotlib figures (plotting part of the preprocessor and postprocessor)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

from .mesh import Mesh
from .solver import Results

CMAP = "jet"


def _edges_collection(coords, elements, **kw):
    segs = coords[elements][:, [0, 1, 2, 0], :]
    return LineCollection(segs, **kw)


def _finish(ax, title):
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.autoscale_view()


def plot_model(mesh: Mesh, F: np.ndarray, BC: np.ndarray, title="Mesh, supports and loads", ax=None):
    """Mesh with nodes coloured by support type and force arrows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6.5))
    ax.add_collection(_edges_collection(mesh.coords, mesh.elements, colors="tab:blue", linewidths=0.5))
    c = mesh.coords
    free = (BC.sum(1) == 0)
    ax.scatter(c[free, 0], c[free, 1], s=6, c="red", label="Nodes", zorder=3)
    bx = (BC[:, 0] == 1) & (BC[:, 1] == 0)
    by = (BC[:, 0] == 0) & (BC[:, 1] == 1)
    bxy = (BC[:, 0] == 1) & (BC[:, 1] == 1)
    ax.scatter(c[bx, 0], c[bx, 1], s=30, c="green", marker=">", label="Fixed x", zorder=4)
    ax.scatter(c[by, 0], c[by, 1], s=30, c="orange", marker="^", label="Fixed y", zorder=4)
    ax.scatter(c[bxy, 0], c[bxy, 1], s=40, c="magenta", marker="s", label="Fixed x,y", zorder=4)
    loaded = np.flatnonzero(np.abs(F).sum(1) > 0)
    if len(loaded):
        Fmax = np.abs(F[loaded]).max()
        scale = 0.12 * np.ptp(mesh.coords[:, 0]) / Fmax
        ax.quiver(c[loaded, 0], c[loaded, 1], F[loaded, 0] * scale, F[loaded, 1] * scale,
                  angles="xy", scale_units="xy", scale=1, color="k", width=0.003,
                  label="Force applied", zorder=5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=8)
    _finish(ax, title)
    return ax.figure


def default_scale(mesh: Mesh, res: Results) -> float:
    umax = np.abs(res.U).max()
    return 0.05 * np.ptp(mesh.coords, axis=0).max() / umax if umax > 0 else 1.0


def plot_deformed(mesh: Mesh, res: Results, scale=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    scale = default_scale(mesh, res) if scale is None else scale
    ax.add_collection(_edges_collection(mesh.coords, mesh.elements, colors="0.8", linewidths=0.4))
    ax.add_collection(_edges_collection(mesh.coords + scale * res.U, mesh.elements,
                                        colors="tab:blue", linewidths=0.6))
    _finish(ax, f"Deformed plate (scale x{scale:.0f})")
    return ax.figure


def _tri(mesh, res, scale):
    xy = mesh.coords if res is None else mesh.coords + scale * res.U
    return Triangulation(xy[:, 0], xy[:, 1], mesh.elements)


def plot_nodal_field(mesh: Mesh, values, title, res: Results | None = None, scale=0.0,
                     label="", ax=None, levels=None):
    """Continuous (interpolated) colouring from nodal values."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    tri = _tri(mesh, res, scale)
    tpc = ax.tripcolor(tri, values, shading="gouraud", cmap=CMAP)
    ax.figure.colorbar(tpc, ax=ax, label=label, shrink=0.85)
    _finish(ax, title)
    return ax.figure


def plot_element_field(mesh: Mesh, values, title, res: Results | None = None, scale=0.0,
                       label="", ax=None):
    """One colour per element with a fringe bar."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    tri = _tri(mesh, res, scale)
    tpc = ax.tripcolor(tri, facecolors=values, cmap=CMAP, edgecolors="none")
    ax.figure.colorbar(tpc, ax=ax, label=label, shrink=0.85)
    _finish(ax, title)
    return ax.figure


def plot_principal(mesh: Mesh, res: Results, which=1, ax=None):
    """Principal stress vectors (two opposite arrows per element)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    s1, s2, phi = res.principal.T
    s = s1 if which == 1 else s2
    ang = phi if which == 1 else phi + np.pi / 2
    cg = mesh.centroids
    h = np.sqrt(np.median(mesh.areas))
    smax = np.abs(s).max() or 1.0
    L = 2.5 * h * np.abs(s) / smax  # arrow length ~ |sigma|
    u, v = L * np.cos(ang) / 2, L * np.sin(ang) / 2
    tension = s >= 0
    ax.add_collection(_edges_collection(mesh.coords, mesh.elements, colors="0.85", linewidths=0.3))
    for sign in (1, -1):
        # tension: arrows point outwards, compression: inwards
        x0 = np.where(tension, cg[:, 0], cg[:, 0] + sign * u)
        y0 = np.where(tension, cg[:, 1], cg[:, 1] + sign * v)
        dx = np.where(tension, sign * u, -sign * u)
        dy = np.where(tension, sign * v, -sign * v)
        q = ax.quiver(x0, y0, dx, dy, s, cmap=CMAP, angles="xy", scale_units="xy", scale=1,
                      width=0.0028, headwidth=3, headlength=3.5, clim=(s.min(), s.max()))
    ax.figure.colorbar(q, ax=ax, label=f"$\\sigma_{which}$ [MPa]", shrink=0.85)
    _finish(ax, f"Principal stress vectors $\\sigma_{which}$")
    return ax.figure


def plot_line_distribution(curves, title="Distribution of $\\sigma_x$ on the symmetry line x = a",
                           ax=None, theory=None):
    """``curves``: list of (label, y, sx)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    for label, y, sx in curves:
        y, sx = np.asarray(y, float), np.asarray(sx, float)
        dy = np.diff(y)
        gap = np.flatnonzero(dy > 4 * np.median(dy)) + 1 if len(dy) else []
        # break the curve across the hole
        y, sx = np.insert(y, gap, np.nan), np.insert(sx, gap, np.nan)
        ax.plot(sx, y, marker=".", label=label)
    if theory is not None:
        ax.axvline(theory, color="k", ls="--", lw=1, label="Theory max")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("$\\sigma_x$ [MPa]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax.figure


def plot_line_arrows(mesh: Mesh, line, ax=None):
    """sigma_x on the symmetry line drawn as arrows over the mesh."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.add_collection(_edges_collection(mesh.coords, mesh.elements, colors="k", linewidths=0.3))
    smax = np.abs(line.sx).max() or 1.0
    scale = 0.35 * np.ptp(mesh.coords[:, 0]) / 2 / smax
    ax.quiver(mesh.coords[line.nodes, 0], line.y, line.sx * scale, 0 * line.sx,
              angles="xy", scale_units="xy", scale=1, color="red", width=0.004)
    _finish(ax, "Distribution of $\\sigma_x$ on x = a")
    return ax.figure
