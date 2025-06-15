from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Tuple

class Transformation(ABC):
    """
    Abstract base class for all 2D frame transformation schemes
    (linear, corotational, etc).
    
    Provides the geometric and kinematic relationships between:
    - global coordinates
    - local (element-aligned) coordinates
    - basic deformation modes
    
    Concrete subclasses must implement all transformation logic.
    """

    @abstractmethod
    def get_length(self) -> float:
        """
        Return the current (possibly deformed) length of the element.
        Used for scaling deformation gradients or geometric stiffness.
        """
        ...
        
    @abstractmethod
    def get_L0(self) -> float:
        """
        Return the undeformed (initial) length of the element.
        Used for reference in geometric transformations.
        """
        ...

    @abstractmethod
    def get_cosine_director(self) -> Tuple[float, float, float]:
        """
        Return the (cosine, sine, length) of the element axis.
        May be from undeformed (linear) or deformed (corotational) configuration.
        
        Returns:
            Tuple[float, float, float]: (c, s, L)
        """
        ...

    @abstractmethod
    def get_Tlg(self) -> ndarray:
        """
        Return the 6×6 transformation matrix from global to local system.

        This matrix maps:
            u_local = Tlg × u_global
        and
            f_global = Tlg.T × f_local
        """
        ...

    @abstractmethod
    def get_Tbl(self) -> ndarray:
        """
        Return the 3×6 transformation matrix from local to basic system.

        This maps:
            u_basic = Tbl × u_local
        and
            f_local = Tbl.T × f_basic
        """
        ...

    @abstractmethod
    def geometric_transformation_matrix(self, P: float) -> ndarray:
        """
        Return the geometric stiffness matrix (6×6) for the current configuration.
        This matrix depends on the axial force P in the element and contributes
        to second-order geometric effects (P–Δ behavior).

        Args:
            P (float): Axial force in the element (positive in tension)

        Returns:
            ndarray: 6×6 geometric stiffness matrix in global coordinates
        """
        ...
