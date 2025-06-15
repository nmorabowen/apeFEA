import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


class Restraints:
    """
    Manages degrees of freedom (DOFs) restraints and single-point (SP) prescribed displacements for a Node.

    Parameters
    ----------
    node : Node
        The node to which the restraints apply.
    restrain_list : list of str, optional
        A list specifying the boundary condition for each DOF:
        - 'r' = restrained (fixed)
        - 'f' = free
        Defaults to all 'f' (free) if not provided.
    restrain_displacement : list of float, optional
        Prescribed displacement values for each DOF.
        Defaults to zero displacement for all restrained DOFs.

    Attributes
    ----------
    restraints : ndarray of str
        Boundary conditions for each DOF ('r' or 'f').
    displacements : ndarray of float
        Prescribed displacements for each DOF.
    """

    def __init__(
        self,
        node: "Node",
        restrain_list: Optional[List[str]] = None,
        restrain_displacement: Optional[List[float]] = None
    ):
        self.node = node

        self.restraints = np.array(restrain_list) if restrain_list else np.array(['f'] * node.ndof)
        self.displacements = np.array(restrain_displacement) if restrain_displacement else np.zeros(node.ndof)

    def apply_BC(self, boundary_condition: List[str]) -> None:
        """
        Apply or update binary boundary conditions for each DOF.

        Parameters
        ----------
        boundary_condition : list of str
            List of 'r' or 'f' for each DOF.

        Raises
        ------
        AssertionError
            If the list length does not match the node's DOF count.
        """
        assert len(boundary_condition) == self.node.ndof, "Boundary condition length mismatch"
        self.restraints = np.array(boundary_condition)

    def apply_SP_displacements(self, SP_displacements: List[float]) -> None:
        """
        Apply or update prescribed displacements for each DOF.

        Parameters
        ----------
        SP_displacements : list of float
            List of prescribed displacements for each DOF.

        Raises
        ------
        AssertionError
            If the list length does not match the node's DOF count.
        """
        assert len(SP_displacements) == self.node.ndof, "Displacement length mismatch"
        self.displacements = np.array(SP_displacements)
