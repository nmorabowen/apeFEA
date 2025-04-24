# apeFEM/core/numbering.py

from .config import FEMConfig

class NodeNumberer:
    """
    Handles global ID and DOF indexing for a set of nodes.
    
    - Assigns a unique external ID to each node if not already set.
    - Assigns internal index used for global DOF assembly.
    """

    def __init__(self):
        self._next_id = 1
        self._next_index = 0

    def number(self, nodes):
        """
        Assign IDs and DOF indices to a list of Node objects.

        Args:
            nodes (list[Node]): List of nodes to be numbered.
        """
        for node in nodes:
            # Assign unique external ID if not set
            if node.id is None:
                node.id = self._next_id
                self._next_id += 1

            # Assign internal index and DOFs
            node.index = self._next_index
            node.set_dof_indices(self._next_index * FEMConfig.nDof)
            self._next_index += 1
