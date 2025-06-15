from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


class Element(ABC):
    """
    Abstract base class for all finite elements.

    Defines the essential API for structural finite elements,
    including stiffness matrix computation in multiple coordinate systems,
    global DOF indexing, and visualization support.

    Attributes
    ----------
    id : int
        Unique element identifier.
    nodes : list
        List of connected node objects.
    """

    def __init__(self, id: int, nodes: List):
        self.id: int = id
        self.nodes: List = nodes

    @abstractmethod
    def get_basic_stiffness_matrix(self) -> ndarray:
        """
        Compute the element's stiffness in the basic (deformation) system.

        Returns
        -------
        ndarray
            Basic stiffness matrix (usually 3×3).
        """
        ...

    @abstractmethod
    def get_local_stiffness_matrix(self) -> ndarray:
        """
        Transform the basic stiffness matrix to the local coordinate system.

        Returns
        -------
        ndarray
            Local stiffness matrix (typically 6×6).
        """
        ...

    @abstractmethod
    def get_global_stiffness_matrix(self) -> ndarray:
        """
        Transform the local stiffness matrix to the global coordinate system.

        Returns
        -------
        ndarray
            Global stiffness matrix (typically 6×6).
        """
        ...

    @abstractmethod
    def get_assembly_stiffness_matrix(self) -> ndarray:
        """
        Return the element stiffness matrix in global coordinates
        for direct assembly into the global system.

        Returns
        -------
        ndarray
            Final stiffness matrix for global K assembly.
        """
        ...

    @abstractmethod
    def _elementIndices(self) -> Tuple[ndarray, ndarray]:
        """
        Return the global DOF indices and their associated restraint flags.

        Returns
        -------
        tuple
            (global_dof_indices, restraint_flags)
        """
        ...

    @abstractmethod
    def plot(self, ax: plt.Axes, **kwargs) -> None:
        """
        Plot the undeformed or deformed shape of the element.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        kwargs : dict
            Passed to matplotlib plotting function (e.g., color, linewidth).
        """
        ...
