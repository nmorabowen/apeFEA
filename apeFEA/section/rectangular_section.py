# apeFEA/sections/rectangular_section.py
from .section import Section

class RectangularSection(Section):
    """
    Rectangular cross-section with width b and height h.
    """

    def __init__(self, b: float, h: float, name="RectangularSection"):
        """
        Args:
            b (float): Base width
            h (float): Height
        """
        super().__init__(name)
        self.b = b
        self.h = h

    def get_area(self):
        """Return cross-sectional area A = b * h"""
        return self.b * self.h

    def get_inertia(self):
        """Return moment of inertia I = b * h^3 / 12"""
        return self.b * self.h**3 / 12

    def __str__(self):
        return f"{self.name} (b={self.b}, h={self.h})"

    def __repr__(self):
        return f"RectangularSection(name='{self.name}', b={self.b}, h={self.h})"
