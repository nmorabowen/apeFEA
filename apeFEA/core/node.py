import numpy as np
from numpy import ndarray
from typing import List, Optional, TYPE_CHECKING

from .restraints import Restraints
from .nodal_load import NodalLoad


class Node:
    """
    Represents a node in the finite element model with degrees of freedom, displacements,
    external/internal forces, restraints, and loads.

    Attributes
    ----------
    id : int
        Unique identifier of the node.
    coords : ndarray
        Coordinates of the node in space.
    ndof : int
        Number of degrees of freedom at this node.
    u_trial : ndarray
        Current trial displacements.
    u_committed : ndarray
        Last committed displacements.
    f_internal : ndarray
        Internal force vector.
    f_external : ndarray
        External force vector.
    idx : ndarray
        Global DOF indices.
    restraints : Restraints
        Restraint information including boundary conditions and prescribed displacements.
    loads : list of NodalLoad
        External nodal loads acting on the node.
    """

    def __init__(
        self,
        id: int,
        coords: List[float],
        restrain_list: Optional[List[str]] = None,
        ndof: int = 3
    ):
        """
        Initialize a Node object.

        Parameters
        ----------
        id : int
            Node identifier.
        coords : list of float
            Coordinates of the node.
        restrain_list : list of str, optional
            Restraint flags ('r' or 'f') per DOF.
        ndof : int
            Number of DOFs for this node.
        """
        self.id: int = id
        self.coords: ndarray = np.array(coords, dtype=float)
        self.ndof: int = ndof

        self.u_trial: ndarray = np.zeros((self.ndof, 1))
        self.u_committed: ndarray = np.zeros((self.ndof, 1))

        self.f_internal: ndarray = np.zeros((self.ndof, 1))
        self.f_external: ndarray = np.zeros((self.ndof, 1))

        self.idx: ndarray = self.set_indices(start_index=self.id)
        self.dof: List[int] = []

        self.restraints = Restraints(node=self, restrain_list=restrain_list)
        self.loads: List[NodalLoad] = []

    # ---------------------------------------------------
    # Boundary Conditions
    def set_restrain_displacements(self, displacements: List[float]) -> None:
        """Apply single-point displacement restraints (SP displacements)."""
        self.restraints.apply_SP_displacements(displacements)

    def set_restraints(self, restraints: List[str]) -> None:
        """Set fixed/free restraints for each DOF ('r' or 'f')."""
        self.restraints.apply_BC(restraints)

    # ---------------------------------------------------
    # Load Handling
    def add_load(self, load: List[float]) -> None:
        """Attach a nodal load to this node."""
        load_object = NodalLoad(self, load)
        self.loads.append(load_object)

    # ---------------------------------------------------
    # State Management
    def commit_state(self) -> None:
        """Save trial displacements as committed state."""
        self.u_committed[:] = self.u_trial

    def reset_trial(self) -> None:
        """Revert trial state to last committed state."""
        self.u_trial[:] = self.u_committed

    def revert_to_start(self) -> None:
        """Zero out all displacements and forces (restart state)."""
        self.u_trial[:] = 0.0
        self.u_committed[:] = 0.0
        self.f_internal[:] = 0.0
        self.f_external[:] = 0.0

    # ---------------------------------------------------
    # Indexing
    def set_indices(self, start_index: int) -> ndarray:
        """Return array of global DOF indices based on node ID."""
        return np.arange(self.ndof) + self.ndof * (self.id - 1)

    # ---------------------------------------------------
    # Getters
    def get_trial_disp(self) -> ndarray:
        return self.u_trial

    def get_committed_disp(self) -> ndarray:
        return self.u_committed

    def get_coords(self) -> ndarray:
        return self.coords

    # ---------------------------------------------------
    # Visualization / Debug
    def plot(self, ax=None, **kwargs) -> None:
        """Plot the node's coordinates using matplotlib."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.coords[0], self.coords[1], 'o', **kwargs)
        ax.text(self.coords[0], self.coords[1], str(self.id), fontsize=12, ha='right', va='bottom')

    def printSummary(self) -> None:
        """Print a summary of this node's key attributes and loads."""
        print(f'--------------------------------------------')
        print(f"Node {self.id} at {self.coords}")
        print(f"Indices: {self.idx}")
        print(f"Restraints: {self.restraints.restraints}")
        if self.loads:
            print(f"Load Pattern:")
            for load in self.loads:
                print(f"  {load}")
        print(f'--------------------------------------------\n')

    def __str__(self):
        return f"Node {self.id} at {self.coords}"

    def __repr__(self):
        return self.__str__()
