from numpy import ndarray
from apeFEA.materials import Material  # adjust if your structure changes


class Section:
    """
    Represents a uniaxial section with a material model and geometric properties.

    Parameters
    ----------
    material : Material
        Uniaxial material model (e.g., LinearElastic, EPP).
    A : float
        Cross-sectional area.
    I : float
        Second moment of area (for bending).

    Methods
    -------
    get_stiffness_matrix() -> tuple[float, float]
        Returns EA and EI values at the current tangent modulus.
    """

    def __init__(self, material: Material, A: float, I: float):
        self.material = material
        self.A = A
        self.I = I

    def get_stiffness_matrix(self) -> tuple[float, float]:
        """
        Compute section axial and flexural stiffness using current tangent modulus.

        Returns
        -------
        EA : float
            Axial stiffness
        EI : float
            Flexural stiffness
        """
        Et = self.material.get_tangent()
        EA = Et * self.A
        EI = Et * self.I
        return EA, EI
