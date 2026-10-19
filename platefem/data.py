"""Problem data: geometry and material of the plate."""

from __future__ import annotations

from dataclasses import dataclass, field

# Default problem: 170 x 100 x 4 mm steel plate with a 42.4 mm central hole.
DEFAULTS = dict(length=170.0, width=100.0, t=4.0, d=42.4, E=210000.0, v=0.3)


@dataclass
class PlateData:
    """Geometry and material of the plate. Units: N, mm, MPa (N/mm^2).

    The plate occupies ``[0, 2a] x [0, 2b]`` and the hole is centred at ``(a, b)``.
    Use :meth:`from_dimensions` to create it from the total length and width.
    """

    t: float = DEFAULTS["t"]  # thickness [mm]
    d: float = DEFAULTS["d"]  # hole diameter [mm]
    E: float = DEFAULTS["E"]  # Young's modulus [MPa]
    v: float = DEFAULTS["v"]  # Poisson ratio [-]
    a: float = DEFAULTS["length"] / 2  # half length of the plate [mm]
    b: float = DEFAULTS["width"] / 2  # half width of the plate [mm]
    G: float = field(init=False)  # shear modulus E / (2 (1 + v))

    def __post_init__(self) -> None:
        for name in ("t", "d", "E", "a", "b"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.v < 0.5:
            raise ValueError("Poisson ratio must be in [0, 0.5)")
        if not self.d < 2 * min(self.a, self.b):
            raise ValueError(
                f"Hole diameter d = {self.d} mm must be smaller than {2 * min(self.a, self.b)} mm"
            )
        self.G = self.E / (2 * (1 + self.v))

    @classmethod
    def from_dimensions(
        cls,
        length: float = DEFAULTS["length"],
        width: float = DEFAULTS["width"],
        t: float = DEFAULTS["t"],
        d: float = DEFAULTS["d"],
        E: float = DEFAULTS["E"],
        v: float = DEFAULTS["v"],
    ) -> "PlateData":
        """Create the plate from its total length (x), total width (y), thickness,
        hole diameter, Young's modulus and Poisson ratio."""
        return cls(t=t, d=d, E=E, v=v, a=length / 2, b=width / 2)

    @property
    def R(self) -> float:
        """Hole radius."""
        return self.d / 2

    @property
    def H(self) -> float:
        """Total plate width (2b)."""
        return 2 * self.b

    @property
    def L(self) -> float:
        """Total plate length (2a)."""
        return 2 * self.a
