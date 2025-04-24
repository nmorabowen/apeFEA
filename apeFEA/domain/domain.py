# apeFEM/core/domain.py

from .domain_node_manager import DomainNodeManager
from ..geometry.geometry import Geometry

class Domain:
    def __init__(self, name:str, geometry:Geometry=None, numberer_type='Plain'):
        self.name = name
        self.geometry = geometry  # shared geometry object
        self.numberer_type = numberer_type
        
        self.nodes:DomainNodeManager = DomainNodeManager(self)  # Call the DomainNodeManager Class to handle nodes creation and management
        

    def print_nodal_info(self):
        print('-----------------------------------------------')
        for node in self.nodes.domain_nodes:
            print(
                f'Geometric ID: {node.node.id}, Domain ID: {node.id_domain}, indices: {node.index}, Nodal Loads: {node.nodal_loads}, Restraints: {node.restrains}, Nodal Displacements: {node.nodal_displacements}'
                )
        print('-----------------------------------------------')
    
    def __repr__(self):
        return f"FEMDomain(name={self.name}, n_nodes={len(self.nodes)})"
