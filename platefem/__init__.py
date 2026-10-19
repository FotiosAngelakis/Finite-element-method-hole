"""platefem - 2D plane-stress FEM analysis of a plate with a central hole.

Constant-strain triangles (CST), automatic mapped meshing with geometric
grading towards the hole, tension and in-plane bending load cases,
stress-concentration (notch) factors and mesh sensitivity studies.
"""

from .data import PlateData
from .mesh import Mesh, generate_mesh, spacing
from .loads import LoadCase, apply_load_case
from .solver import Results, solve
from .postprocess import nodal_average, symmetry_line_stress, notch_factor

__all__ = [
    "PlateData",
    "Mesh",
    "generate_mesh",
    "spacing",
    "LoadCase",
    "apply_load_case",
    "Results",
    "solve",
    "nodal_average",
    "symmetry_line_stress",
    "notch_factor",
]

__version__ = "1.0.0"
