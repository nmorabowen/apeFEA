# apeFEA/core/line.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Line:
    """
    Represents a line between two nodes in the geometry.

    Attributes:
        id (int): Unique identifier for the line.
        node_start (Node): Starting node of the line.
        node_end (Node): Ending node of the line.
    """

    def __init__(self, node_start, node_end, id=None):
        self.node_start = node_start
        self.node_end = node_end
        self.id = id

        if node_start.id == node_end.id:
            raise ValueError("A line must connect two distinct nodes.")

        logger.debug(f"Created Line(id={self.id}) from Node {node_start.id} to Node {node_end.id}")

    def __repr__(self):
        return f"Line(id={self.id}, start={self.node_start.id}, end={self.node_end.id})"
    
    def get_length(self):
        """Returns the Euclidean distance between the two nodes.
        This parameter is not computed as attributes because they may change if the nodes are moved.
        """
        return np.linalg.norm(self.node_end.coords - self.node_start.coords)
    
    def get_angle(self):
        """
        Returns the angle of the line with respect to the global X-axis.
        This parameter is not computed as attributes because they may change if the nodes are moved.
        Returns:
            dict: {'radians': float, 'degrees': float}
        """
        delta = self.node_end.coords - self.node_start.coords

        if len(delta) != 2:
            raise NotImplementedError("Angle computation is only implemented for 2D lines.")

        theta_rad = np.arctan2(delta[1], delta[0])
        theta_deg = np.degrees(theta_rad)

        return {"radians": theta_rad, "degrees": theta_deg}
