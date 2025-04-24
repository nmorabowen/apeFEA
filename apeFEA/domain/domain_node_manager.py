# apeFEM/domain/domain_node_manager.py

from .node_domain import NodeDomain
from .numbering import Numberer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DomainNodeManager:
    
    def __init__(self, domain):
        self.domain=domain
        self.ids={
            'geometry':set(),
            'domain':set()
        }
    
        # Get the nodes object list and element object list from the geometry
        self.geometric_nodes = self.domain.geometry.get_nodes_list()
        self.domain_nodes=self._create_node_domain() # Create node domains from geometric nodes
        self._create_domain_ids() # Create the domain ids for the nodes
        self._set_nodes_indices() # Set the domain ids for the nodes
        # Create a geometric mapping for the domain node objects to aid in the node properties instanciation
        self.geometric_node_mapping=self._create_geometric_node_mapping()
        

        
    def _create_domain_ids(self):
        numberer=Numberer(nodes_array=self.domain_nodes,
                          type=self.domain.numberer_type)
        numberer.number()
        
    def _set_nodes_indices(self):
        for node in self.domain_nodes:
            node.set_index(node.id_domain)
            
    def _create_node_domain(self):
        """method to create node domains from the geometric nodes.
        """
        domain_nodes = []
        for node in self.geometric_nodes:
            # Create a NodeDomain object for each geometric node
            node_domain = NodeDomain(node=node)
            domain_nodes.append(node_domain)
            
        print(f'Created {len(domain_nodes)} NodeDomain objects.')
        return domain_nodes
    
    def _create_geometric_node_mapping(self):
        geometric_node_mapping = {}
        for node in self.domain_nodes:
            geometric_node_mapping[node.node.id] = node
        return geometric_node_mapping
    
    def get_node_by_geometric_id(self, id:int):
        if id in self.geometric_node_mapping:
            return self.geometric_node_mapping[id]
        else:
            raise ValueError(f"Node with ID {id} not found in the geometric mapping.")
    
    def set_nodal_load(self, geometric_id: int, load: list):
        """
        Set the nodal load for a domain node based on its geometric node ID.

        Args:
            geometric_id (int): The ID of the geometric node.
            load (list): The nodal load vector (must match FEMConfig.nDof).

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If the node ID does not exist.
        """
        if not isinstance(load, list):
            raise TypeError("Load must be a list.")
        if not isinstance(geometric_id, int):
            raise TypeError("Geometric ID must be an integer.")
        if len(load) != 3:  # Assuming 3D load vector
            raise ValueError("Load vector must have length 3.")
        
        domain_node = self.get_node_by_geometric_id(geometric_id)
        domain_node.nodal_loads = load
        logger.info(f"Set nodal load {load} on domain node with geometric ID {geometric_id}")

    def set_restrains(self, geometric_id: int, restrains: list):
        """
        Set the restrains for a domain node based on its geometric node ID.

        Args:
            geometric_id (int): The ID of the geometric node.
            restrains (list): The restrains vector (must match FEMConfig.nDof).

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If the node ID does not exist.
        """
        if not isinstance(restrains, list):
            raise TypeError("Restrains must be a list.")
        if not isinstance(geometric_id, int):
            raise TypeError("Geometric ID must be an integer.")
        if len(restrains) != 3:
            raise ValueError("Restrains vector must have length 3.")
        for val in restrains:
            if val not in ("r", "f"):
                raise ValueError("Each entry in restrains must be either 'r' (restrained) or 'f' (free).")
        
        
        domain_node = self.get_node_by_geometric_id(geometric_id)
        domain_node.restrains = restrains
        logger.info(f"Set restrains {restrains} on domain node with geometric ID {geometric_id}")
        
        
        