# apeFEA/materials/material.py
import numpy as np
from abc import ABC, abstractmethod

class Material(ABC):
    """
    Base abstract class for all material models.
    
    This class defines the interface that all material models must implement,
    regardless of whether they're linear or nonlinear, isotropic or anisotropic.
    """
    
    def __init__(self, name=None):
        """
        Initialize material with a name for identification.
        
        Args:
            name (str, optional): Material identifier
        """
        self.name = name
        self._strain_history = []  # For materials that need to track history
    
    @abstractmethod
    def get_stress(self, strain):
        """Return stress vector from input strain vector."""
        pass
    
    @abstractmethod
    def get_tangent_modulus(self, strain):
        """Return constitutive matrix (Jacobian) at given strain."""
        pass
    
    def update_state(self, strain):
        """Track strain history (for nonlinear/hysteretic materials)."""
        self._strain_history.append(np.copy(strain))
    
    def reset_state(self):
        """Reset the material state to its initial condition."""
        self._strain_history = []