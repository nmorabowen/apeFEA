import numpy as np
from numpy import ndarray

from apeFEA.core.node import Node
from apeFEA.elements.one_dimension.frame_element import FrameElement
from apeFEA.timeseries.timeseries_abstraction import TimeSeries
from apeFEA.timeseries.timeseries import LinearRampTimeSeries



class Model:
    """
    The `Model` class represents a finite element system composed of nodes, elements, 
    and external loads. It manages the global assembly of the stiffness matrix, 
    internal resisting forces, external loads, and time-dependent loading via a 
    time series. It also maintains trial, committed, and reset states across 
    nonlinear iterations.

    Attributes:
        elements (list[FrameElement]): List of frame elements in the model.
        timeseries (TimeSeries): Time-dependent scaling function for loads.
        ndof (int): Number of degrees of freedom per node (default: 3).
        nodes (list[Node]): Unique list of all nodes in the model.
        number_of_nodes (int): Total number of nodes.
        number_of_elements (int): Total number of elements.
        system_ndof (int): Total number of DOFs in the global system.
        free_indices (ndarray): Indices of free DOFs.
        restrained_indices (ndarray): Indices of restrained DOFs.

    Methods:
        get_resistance_force(): Assembles global internal resisting force vector.
        get_external_force(t): Computes external force vector at pseudotime t.
        calculate_residual(t): Returns residual vector R = F_ext - F_int at time t.
        residual_norm(t, norm_type): Returns norm (L2 or inf) of the residual.
        update_trial_state(u): Updates nodal trial states with displacement vector u.
        commit_state(): Commits current trial state for all nodes and elements.
        reset_trial(): Resets all trial states to last committed state.
        revert_to_start(): Reverts all states to initial zero configuration.
        get_stiffness_matrix(): Assembles and returns the global tangent stiffness matrix.
        _assemble_displacement_vector_committed(): Gathers u_committed from all nodes.
        print_trial_committed_state(): Prints trial and committed states of all nodes.
        print_summary(): Prints a structural summary of the model setup.
    """
    def __init__(self, 
                 elements: list[FrameElement], 
                 timeseries: TimeSeries = LinearRampTimeSeries, 
                 ndof: int = 3, 
                 print_summary: bool = False):
        
        self.elements = elements
        self.timeseries = timeseries()
        self.ndof = ndof
        
        # Get the list of nodes from elements
        self.nodes = self._get_nodes_list()

        # Get info for assembly
        self.number_of_nodes = len(self.nodes)
        self.number_of_elements = len(elements)
        self.system_ndof = self.number_of_nodes * self.ndof
        self.free_indices, self.restrained_indices = self._get_mapping_indices()
        
        if print_summary:
            self.print_summary()

    def _get_nodes_list(self) -> list[Node]:
        return list({node for element in self.elements for node in element.nodes})
    
    def _get_restrained_indices(self) -> np.ndarray:
        nodeIndex = np.empty(self.system_ndof, dtype=str)
        for node in self.nodes:
            nodeIndex[node.idx] = node.restraints.restraints 
        return nodeIndex
    
    def _get_mapping_indices(self) -> list[int]:
        """Get the free and restrain mapping indices for the model.

        Returns:
            list[int]: _description_
        """
        nodeIndex=self._get_restrained_indices()
        free_indices = np.where(nodeIndex == 'f')[0]
        restrained_indices = np.where(nodeIndex == 'r')[0]
        
        return free_indices, restrained_indices
    
    def get_resistance_force(self) -> np.ndarray:
        """Calculate the resistance force vector for a given load vector u_trial.
        This can olnly be evaluated at the trial state

        Returns:
            np.ndarray: _description_
        """
        Fr= np.zeros((self.system_ndof, 1))
        for element in self.elements:
            idx = element.idx
            forces, _ =element.force_recovery()
            Fr[idx] += forces
            
        return Fr

    def get_external_force(self, t:float) -> ndarray:
        """
        Assemble the external force vector at pseudotime `t`.

        Args:
            t (float): Current pseudo-time (used for scaling loads)

        Returns:
            ndarray: External global force vector of shape (system_ndof, 1)
        """
        lam=self.timeseries.get_factor(t)
        Fe = np.zeros((self.system_ndof, 1))
        for node in self.nodes:
            for load in node.loads:
                f = lam * load.load_pattern 
                Fe[node.idx] += f.reshape((3, 1))
        return Fe

    def calculate_residual(self, t: float) -> np.ndarray:
        """
        Compute the global residual force vector at time `t`.

        The residual is:
            R(t) = F_ext(t) - F_int(u_trial)

        Where:
        - F_ext(t) is the external force vector, scaled by time series.
        - F_int is the internal resisting force, computed from current trial state.

        Args:
            t (float): Current pseudo-time (or real time in dynamics)

        Returns:
            np.ndarray: Residual force vector with shape (system_ndof, 1)
        """
        
        return self.get_external_force(t) - self.get_resistance_force()
        
    def residual_norm(self, t: float, norm_type: str = 'L2') -> float:
        R = self.calculate_residual(t)
        R_free = R[self.free_indices]  # Only consider free DOFs
        if norm_type == 'L2':
            return np.linalg.norm(R_free)
        elif norm_type == 'inf':
            return np.linalg.norm(R_free, ord=np.inf)
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
        
    def update_trial_state(self, u: ndarray, verbose:bool = False) -> None:
        """Update the trial state with a global assembly displacement vector.
        This goes node by node mapping the trial displacements

        Args:
            u (ndarray): _description_
        """
        for node in self.nodes:
            node.u_trial[:, 0] = u[node.idx, 0]

            if verbose:
                print('--------------------------------------')
                print(node)
                print(f'Previous Trial Displacement: {node.u_trial.flatten()}')
                print(f'set Trial Displacement: {node.u_trial.flatten()}')
                print('---------------------------------------')
    
    def commit_state(self):
        for node in self.nodes:
            node.commit_state()
        for element in self.elements:
            element.commit_state()
            
    def reset_trial(self):
        for node in self.nodes:
            node.reset_trial()
        for element in self.elements:
            element.reset_trial()

    def revert_to_start(self):
        for node in self.nodes:
            node.revert_to_start()
        for element in self.elements:
            element.revert_to_start()
    
    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Assemble the global tangent stiffness matrix K for the model at the trial state.
        """
        
        K = np.zeros((self.number_of_nodes * self.ndof, self.number_of_nodes * self.ndof))
        
        for element in self.elements:
            Ke=element.get_assembly_stiffness_matrix()
            idx=element.idx
            K[np.ix_(idx, idx)] += Ke
            
        return K
    
    def _assemble_displacement_vector_committed(self) -> ndarray:
        """
        Assemble the committed displacement vector from all nodes.
        This is used to get the committed state of the model.
        """
        u = np.zeros((self.system_ndof, 1))
        for node in self.nodes:
            u[node.idx] = node.u_committed  # Assumes (3, 1) shape
        return u
    
    def print_trial_committed_state(self) -> None:
        for node in self.nodes:
            print(f"Node {node.id} Trial Displacement: {node.u_trial.flatten()}")
            print(f"Node {node.id} Committed Displacement: {node.u_committed.flatten()}")  
            
    def print_summary(self) -> None:
        """
        Print a summary of the model, including nodes and elements.
        """
        print('=================================================')
        
        print(f"Model Summary:")
        print(f"Number of Nodes: {self.number_of_nodes}")
        print(f"Number of Elements: {self.number_of_elements}")
        print(f"System DOF: {self.system_ndof}\n")
        
        print(f'Model Restraints:')
        print(f"Node Indices: {self._get_restrained_indices()}\n")
        
        print(f'The Mapping Indices:')
        print(f"Free Indices: {self.free_indices}")
        print(f"Restrained Indices: {self.restrained_indices}\n")
        
        print('==================================================')
        
        for node in self.nodes:
            node.printSummary()
        
        for element in self.elements:
            print(element)
    
    

