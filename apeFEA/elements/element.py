# apeFEA/elements/element.py
from abc import ABC, abstractmethod

class Element(ABC):
    """
    Abstract base class for finite elements.
    """

    def __init__(self, id=None, nodes=None, material=None, section=None):
        self.id = id
        self.nodes = nodes  # Lista de nodos (generalmente 2 para barra o viga)
        self.material = material
        self.section = section

    @abstractmethod
    def get_local_stiffness_matrix(self):
        """Return the local (basic) stiffness matrix in element coordinates."""
        pass

    @abstractmethod
    def get_transformation_matrix(self):
        """Return the transformation matrix from local to global coordinates."""
        pass

    def get_global_stiffness_matrix(self):
        """Compute global stiffness matrix as Tᵀ * k_local * T"""
        T = self.get_transformation_matrix()
        k_local = self.get_local_stiffness_matrix()
        return T.T @ k_local @ T

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id})"
