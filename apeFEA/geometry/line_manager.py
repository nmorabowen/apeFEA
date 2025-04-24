# apeFEM/geometry/line_manager.py

from ..core.line import Line
from ..core.node import Node
import logging

logger = logging.getLogger(__name__)

class LineManager:
    
    def __init__(self, geometry):

        self.geometry = geometry
        self.lines_array = []
        self._used_ids = set()
        self._lines_dict = {}
        
    def create_line(self, node_j: Node = None, node_k: Node = None,
                    j_id: int = None, k_id: int = None, id: int = None) -> Line:
        """
        Create a Line object and add it to the geometry.

        You must provide node_j and node_k (as objects), or j_id and k_id (as node IDs).
        ID must be explicitly provided.

        Args:
            node_j (Node, optional): Start node object.
            node_k (Node, optional): End node object.
            j_id (int, optional): Start node ID.
            k_id (int, optional): End node ID.
            id (int): Required Line ID.

        Raises:
            ValueError: If nodes are not found, identical, or ID already exists.
            TypeError: If input types are incorrect.

        Returns:
            Line: Line object created and added to the geometry.
        """

        # Ensure a line ID is provided
        if id is None or not isinstance(id, int):
            raise TypeError("You must provide an integer ID for the line.")

        if id in self._used_ids:
            raise ValueError(f"Line ID {id} is already in use in the {self.geometry.name} domain.")

        # Resolve node objects if only IDs are provided
        if node_j is None:
            if j_id is None:
                raise ValueError("Must provide either node_j or j_id.")
            node_j = self.geometry.nodes.get_node_by_id(j_id)

        if node_k is None:
            if k_id is None:
                raise ValueError("Must provide either node_k or k_id.")
            node_k = self.geometry.nodes.get_node_by_id(k_id)

        if node_j is None or node_k is None:
            raise ValueError("One or both nodes could not be resolved from IDs.")

        if not isinstance(node_j, Node) or not isinstance(node_k, Node):
            raise TypeError("node_j and node_k must be of type Node.")

        if node_j.id == node_k.id:
            raise ValueError("A line must connect two distinct nodes.")

        # Create and register the line
        line = Line(node_j, node_k, id)
        self._used_ids.add(id)
        self.lines_array.append(line)
        self._lines_dict[id] = line

        return line

    def add_line(self, line: Line) -> None:
        """Add a line object to the geometry.

        Args:
            line (Line): Line object to be added.

        Raises:
            ValueError: ValueError if the line ID is already in use in the geometry domain.
        """
        
        if not isinstance(line, Line):
            raise TypeError("Line must be an instance of Line.")
        
        if line.id in self._used_ids:
            raise ValueError(f"Line ID {line.id} is already in use in the {self.geometry.name} domain.")
        
        self._used_ids.add(line.id)
        self.lines_array.append(line)
        self._lines_dict[line.id] = line

    def get_line_by_id(self, id: int) -> Line:
        """Method to return the line with the specified ID.

        Args:
            id (int): Line object ID.

        Raises:
            ValueError: ValueError if the line ID is not found in the geometry domain.
            TypeError: TypeError if the ID is not an integer.

        Returns:
            Line: Line object with the specified ID.
        """
        
        if isinstance(id, int):
            raise TypeError("ID must be an integer.")
        
        try:
            return self._lines_dict[id]
        except KeyError:
            raise ValueError(f"Line ID {id} not found in the {self.geometry.name} domain.")
            return None

    def remove_line(self, line: Line) -> None:
        """Method to remove a line from the geometry.

        Args:
            line (Line): Line object to be removed.

        Raises:
            TypeError: TypeError if the line is not an instance of Line.
        """
        
        if not isinstance(Line):
            raise TypeError("Line must be an instance of Line.")
        
        if line in self.lines_array:
            self.lines_array.remove(line)
            self._used_ids.discard(line.id)
            self._lines_dict.pop(line.id, None)
        else:
            logger.warning(f"Tried to remove line {line} but it was not found in geometry '{self.geometry.name}'.")
            print(f"Tried to remove line {line} but it was not found in geometry '{self.geometry.name}'.")

    def modify_line(self, id:int, new_id:int, line:Line):
        
        if id is None and new_id is None:
            raise ValueError("Either id or new_id must be provided.")
        if not isinstance(int):
            raise TypeError("ID must be an integer.")
        if not isinstance(int):
            raise TypeError("New ID must be an integer.")
        if not isinstance(line, Line):
            raise TypeError("Line must be an instance of Line.")
        
        if id in self._used_ids:
            raise ValueError(f"Line ID {new_id} is already in use in the {self.geometry.name} domain.")
        if not self._used_ids:
            raise ValueError(f"Line ID {id} not found in the {self.geometry.name} domain.")
        
        self._used_ids.discard(id)
        self._used_ids.add(new_id)
        self._lines_dict.pop(id, None)
        self._lines_dict[new_id] = line

    def print_lines_info(self) -> None:
        """Print a summary of the lines in the geometry.
        """
        
        print('----------------------------------------------------------------')
        print(f'The geometry {self.geometry.name} has: {len(self.lines_array)} lines:')
        print('The lines objects are: \n')
        for line in self.lines_array:
            print(f'Line ID: {line.id}, connect Node {line.node_start.id} to Node {line.node_end.id}')
            print(f'Line length: {line.get_length()}')
            print(f'Line angle: {line.get_angle()}')
        print('----------------------------------------------------------------')
        print('\n')