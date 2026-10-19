"""Using platefem as a library (no files, no prompts)."""

import matplotlib.pyplot as plt

from platefem import LoadCase, apply_load_case, calc_data, generate_mesh, notch_factor, solve
from platefem import plots
from platefem.postprocess import recover_nodal

plate = calc_data(5678)  # d = 42.4 mm
load = LoadCase("tension", 2000.0)  # or LoadCase("bending", 10000.0)

mesh = generate_mesh(plate.a, plate.b, plate.d, nox=30, noy=20, ratio_x=0.93, ratio_y=1.0)
F, BC = apply_load_case(mesh, plate, load)
res = solve(mesh, F, BC, plate.E, plate.v, plate.t)

nodal = recover_nodal(mesh, plate, res)
print(notch_factor(mesh, plate, load, res, nodal).summary())

plots.plot_nodal_field(mesh, nodal.stresses[:, 0], "sigma_x (nodal)", label="[MPa]")
plots.plot_principal(mesh, res, which=1)
plt.show()
