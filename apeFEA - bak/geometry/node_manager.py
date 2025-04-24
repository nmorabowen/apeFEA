# apeFEM/geometry/NodeManager.py

import numpy as np
from ..core.node import Node
import logging

logger = logging.getLogger(__name__)

class NodeManager:
    """
    Manages the collection of nodes within a Geometry object.

    This class handles the creation, modification, retrieval, and deletion of nodes,
    as well as internal bookkeeping of node IDs to ensure uniqueness within the geometry domain.

    Attributes:
        geometry (Geometry): The parent Geometry object to which the nodes belong.
        nodes_array (list of Node): Ordered list of Node objects in the geometry.
        _used_ids (set): Set of used node IDs to prevent duplicates.
        _node_dict (dict): Dictionary mapping node IDs to Node objects for fast access.

    Methods:
        create_node(coords, id): Create a new node with specified coordinates and ID.
        add_node(node): Add an externally created Node object.
        get_node_by_id(id): Retrieve a Node by its ID.
        remove_node(node): Remove a node from the geometry.
        modify_node(id or node, new_coords, new_id): Modify the coordinates or ID of a node.
        export_nodes_to_array(): Return the coordinates of all nodes as a NumPy array.
        print_nodes_info(): Print a human-readable list of all nodes in the geometry.
    """

    def  __init__(self, geometry):
        self.nodes_array=[]
        self.geometry=geometry
        self._used_ids = set()
        self._node_dict = {}
    
    def create_node(self, coords: list, id: int = None) -> Node:
        """
        Create and add a new node to the geometry.

        Args:
            coords (list): Coordinates of the node (length must match FEMConfig.dimension).
            id (int, optional): Optional ID for the node. If not provided, an auto-generated ID will be assigned.

        Returns:
            Node: The created Node object.
        """
        
        # Ensure the ids are consistent in the geometry domain
        if id in self._used_ids:
            raise ValueError(f"Node ID {id} is already in use in the {self.geometry.name} domain.")
        
        node = Node(coords, id)
        self._used_ids.add(id)
        self.nodes_array.append(node)
        self._node_dict[id] = node
        
        return node

    def add_node(self, node: Node):
        """
        Add an existing node to the geometry.

        Args:
            node (Node): A Node object to add.
        """
        
        if node.id in self._used_ids:
            raise ValueError(f"Node ID {node.id} is already in use in the {self.geometry.name} domain.")
        
        self.nodes_array.append(node)
        self._node_dict[node.id] = node
        
    def get_node_by_id(self, id: int) -> Node:
        """Return the node with the specified ID.
        Args:
            id (int): _description_

        Returns:
            Node: The node object
        """
        try:
            return self._node_dict[id]
        except KeyError:
            logger.warning(f"Tried to access node with ID {id} but it was not found in geometry '{self.geometry.name}'.")
            print(f"Tried to access node with ID {id} but it was not found in geometry '{self.geometry.name}'.")    
            return None

    def remove_node(self, node: Node):
        """
        Remove a node from the geometry, if it exists.

        Args:
            node (Node): The node to remove.

        Logs:
            Warning if the node is not found.
        """
        if node in self.nodes_array:
            self.nodes_array.remove(node)
            self._used_ids.discard(node.id)
            self._node_dict.pop(node.id, None)
        else:
            logger.warning(f"Tried to remove node {node} but it was not found in geometry '{self.geometry.name}'.")

    def modify_node(self, id:int = None, node: Node = None, new_coords: list = None, new_id: int = None):
        """
        Modify a node’s coordinates and/or ID if it exists in the geometry.

        Args:
            node (Node): The node to modify.
            new_coords (list, optional): New coordinates to assign to the node.
            new_id (int, optional): New ID to assign to the node.

        Logs:
            Warning if node is not found, coordinates do not match dimension, or ID is already in use.
        """
        
        if id is not None and node is not None:
            raise ValueError("Specify either 'id' or 'node', not both.")

        if id is None and node is None:
            raise ValueError("You must specify either 'id' or 'node'.")
        
        if node is None:
            node = self.get_node_by_id(id)
        
        if node not in self.nodes_array:
            logger.warning(f"Tried to modify node {node} but it was not found in geometry '{self.geometry.name}'.")
            return

        if new_coords is not None:
            if len(new_coords) != len(node.coords):
                logger.warning(f"New coordinates {new_coords} don't match node dimension.")
                raise ValueError(f"New coordinates must match the node's dimension ({len(node.coords)}D).")
            else:
                node.coords = new_coords
                logger.info(f"Updated coordinates of Node(id={node.id}) to {new_coords}")

        if new_id is not None and new_id != node.id:
            if new_id in self._used_ids:
                logger.warning(f"Cannot assign new ID {new_id}; already in use in geometry '{self.geometry.name}'.")
                raise ValueError(f"Node ID {new_id} is already in use in the {self.geometry.name} domain.")
            else:
                old_id = node.id
                node.id = new_id
                self._used_ids.discard(old_id)
                self._used_ids.add(new_id)
                self._node_dict.pop(old_id)
                self._node_dict[new_id] = node
                logger.info(f"Updated ID of Node from {old_id} to {new_id}")

    def export_nodes_to_array(self) -> np.ndarray:
        """Get an array of all node coordinates.

        Returns:
            np.ndarray: nodal coordinate array
        """
        return np.array([node.coords for node in self.nodes_array])

    def print_nodes_info(self):
        """
        Print information about all nodes in the geometry.
        """
        print('--------------------------------------------------------------')
        print(f'The "{self.geometry.name}" geometry has: {len(self.nodes_array)} nodes.')
        print('This are the nodes: \n')
        for node in self.nodes_array:
            print(f"Node ID: {node.id}, Coordinates: {node.coords}")
        print('--------------------------------------------------------------')
        print('\n')

