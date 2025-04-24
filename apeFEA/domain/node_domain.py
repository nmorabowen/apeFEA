# apeFEM/domain/node_domain.py

from ..core.config import FEMConfig
from ..core.node import Node
import numpy as np

class NodeDomain:
    
    def __init__(self, 
                 node: Node,
                 nodal_loads=None, 
                 nodal_displacements=None,
                 restrains=None):
        """
        Domain-level representation of a Node with loads, displacements, and boundary conditions.
        """

        if not isinstance(node, Node):
            raise TypeError("`node` must be an instance of core.node.Node")
        self.node = node

        # Initialize default arrays with zeros or 'f' for fixed/free if not provided
        default_loads = np.zeros(FEMConfig.nDof)
        default_disps = np.zeros(FEMConfig.nDof)
        default_restrains = np.array(['f'] * FEMConfig.nDof)

        # Use the provided values or defaults
        self.nodal_loads = np.array(nodal_loads) if nodal_loads is not None else default_loads
        self.nodal_displacements = np.array(nodal_displacements) if nodal_displacements is not None else default_disps
        self.restrains = np.array(restrains) if restrains is not None else default_restrains

        # Type checking
        if self.nodal_loads.ndim != 1 or self.nodal_loads.size != FEMConfig.nDof:
            raise ValueError(f"`nodal_loads` must be a 1D array of length {FEMConfig.nDof}")
        if self.nodal_displacements.ndim != 1 or self.nodal_displacements.size != FEMConfig.nDof:
            raise ValueError(f"`nodal_displacements` must be a 1D array of length {FEMConfig.nDof}")
        if self.restrains.ndim != 1 or self.restrains.size != FEMConfig.nDof:
            raise ValueError(f"`restrains` must be a 1D array of length {FEMConfig.nDof}")

        # Set domain ID placeholder
        self.id_domain = None
        self.index=None
        

    def set_domain_id(self, id_domain:int):
        """
        Set the domain id for this node domain.
        
        Parameters:
            id_domain (int): The domain id to set.
        """
        if not isinstance(id_domain, int):
            raise TypeError("id_domain must be an integer")
        self.id_domain = id_domain
        
    def get_dof_indices(self, id_domain) -> list:
        """
        Returns the global DOF indices for this node based on its domain ID.

        Returns:
            list[int]: A list of global DOF indices.
        """
        
        if not isinstance(id_domain, int):
            raise TypeError("id_domain must be an integer")
        return list(np.arange(FEMConfig.nDof) + id_domain * FEMConfig.nDof)

    def set_index(self, id_domain):
        idx=self.get_dof_indices(id_domain)
        self.index=idx
    
    def set_nodal_loads(self, nodal_loads:list):
        """
        Set the nodal loads for this node domain.
        
        Parameters:
            nodal_loads (ndarray): The nodal loads to set.
        """
        if not isinstance(nodal_loads, list):
            raise TypeError("nodal_loads must be a numpy array")
        if len(nodal_loads) != FEMConfig.nDof:
            raise ValueError(f"nodal_loads must have length {FEMConfig.nDof}")
        
        self.nodal_loads = np.array(nodal_loads)
        
    def set_nodal_displacements(self, nodal_displacements:list):
        """
        Set the nodal displacements for this node domain.
        
        Parameters:
            nodal_displacements (ndarray): The nodal displacements to set.
        """
        if not isinstance(nodal_displacements, list):
            raise TypeError("nodal_displacements must be a numpy array")
        if len(nodal_displacements) != FEMConfig.nDof:
            raise ValueError(f"nodal_displacements must have length {FEMConfig.nDof}")
        
        self.nodal_displacements = np.array(nodal_displacements)
        
    def set_restrains(self, restrains:list):
        """
        Set the restrains for this node domain.
        
        """
        if not isinstance(restrains, list):
            raise TypeError("restrains must be a numpy array")
        if len(restrains) != FEMConfig.nDof:
            raise ValueError(f"restrains must have length {FEMConfig.nDof}")
        
        self.restrains = np.array(restrains)