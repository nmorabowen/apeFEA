# apeFEM/core/domain.py

from .config import FEMConfig

class Domain:
    def __init__(self, name="global", geometry=None):
        self.name = name
        self.geometry = geometry  # shared geometry object
        self.nodes = OrderedDict()  # domain-local view of nodes
        self.elements = []          # list of Element objects
        self._dof_counter = 0

        if geometry:
            self.use_geometry(geometry)

    def use_geometry(self, geometry):
        for node_id, node in geometry.nodes.items():
            self.nodes[node_id] = node  # references the *same* Node object

    def assign_dofs(self):
        self._dof_counter = 0
        for i, node in enumerate(self.nodes.values()):
            node.index[self.name] = i
            node.dofs[self.name] = node.get_dof_indices(self._dof_counter)
            self._dof_counter += FEMConfig.nDof

    def apply_constraint(self, node_id, dof_idx):
        self.nodes[node_id].constraints[self.name][dof_idx] = True

    def apply_force(self, node_id, dof_idx, value):
        self.nodes[node_id].forces[self.name][dof_idx] += value

    def __repr__(self):
        return f"FEMDomain(name={self.name}, n_nodes={len(self.nodes)})"
