# apeFEM/core/node.py
import logging
import numpy as np
from .config import FEMConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # You can switch to DEBUG for more verbosity

class Node:
    """
    Represents a node in the FEM domain with coordinates and degrees of freedom.
    """

    def __init__(self, coords: list, id: int = None):
        
        assert len(coords) == FEMConfig.dimension, \
            f"Coordinates must match problem dimension ({FEMConfig.dimension}D)"
        
        if not isinstance(coords, list):
            raise TypeError("Coordinates must be a list")
        
        self.coords = np.array(coords)
        self.id=id

        logger.debug(f"Created Node {self.id} at {self.coords}")

    def get_dof_indices(self, domain_start_index: int) -> list:
        """
        Given a starting index from the domain, returns what the DOF indices
        for this node would be without setting them. This is useful for domain-level
        numbering without mutating the node.

        Parameters:
            domain_start_index (int): The starting index provided by the domain.

        Returns:
            list[int]: A list of DOF indices for this node.
        """
        return np.array(range(domain_start_index, domain_start_index + FEMConfig.nDof))


    def __repr__(self):
        return f"Node(id={self.id}, coords={self.coords})"
