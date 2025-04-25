# apeFEA/materials/uniaxial_elastic_material.py
import numpy as np
from .material import Material

class UniaxialElasticMaterial(Material):
    """
    Uniaxial linear elastic material.

    σ = E * ε
    """

    def __init__(self, E: float, name="UniaxialElastic"):
        """
        Args:
            E (float): Young's modulus (modulus of elasticity)
            name (str): Optional name identifier
        """
        super().__init__(name)
        self.E = E

    def get_stress(self, strain: float) -> float:
        """
        Compute stress σ = E * ε

        Args:
            strain (float): Axial strain

        Returns:
            float: Axial stress
        """
        return self.E * strain

    def get_tangent_modulus(self, strain: float = None) -> float:
        """
        For linear elastic, tangent modulus is constant.

        Args:
            strain (float): Strain input (unused)

        Returns:
            float: Tangent modulus (E)
        """
        return self.E
    
    def __str__(self):
        return f"{self.name} (E={self.E})"

    def __repr__(self):
        return f"(name='{self.name}', E={self.E})"

