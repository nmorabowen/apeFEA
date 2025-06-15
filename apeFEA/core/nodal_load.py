import numpy as np
from numpy import ndarray
from typing import List

from apeFEA.core.node import Node  # or use Protocol from TYPE if needed


class NodalLoad:
    """
    Represents a load pattern applied to a specific node.

    Parameters
    ----------
    node : Node
        The node on which the load is applied.
    load_pattern : list of float
        The force components to apply at the node [Fx, Fy, Mz], etc.

    Attributes
    ----------
    node : Node
        Target node where the load is applied.
    load_pattern : ndarray
        Array of force values for each degree of freedom.
    """

    def __init__(self, node: Node, load_pattern: List[float]):
        self.node: Node = node
        self.load_pattern: ndarray = np.array(load_pattern, dtype=float)

    def add_load(self, load_pattern: List[float]) -> None:
        """
        Update the load vector applied to the node.

        Parameters
        ----------
        load_pattern : list of float
            New load values per DOF.

        Raises
        ------
        ValueError
            If the load vector length does not match the node's DOFs.
        """
        if len(load_pattern) != self.node.ndof:
            raise ValueError("Load pattern length must match node's degrees of freedom")
        self.load_pattern = np.array(load_pattern, dtype=float)

    def __str__(self) -> str:
        return f"Load @ Node {self.node.id}: {self.load_pattern}"

    def __repr__(self) -> str:
        return f"NodalLoad(node={self.node.id}, load={self.load_pattern.tolist()})"

    def plot(self, ax, scale: float = 1.0, **kwargs) -> None:
        """
        Plot the nodal load as a vector arrow using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which to draw the load vector.
        scale : float, optional
            Scale factor to visually enlarge the force arrow (default is 1.0).
        kwargs : dict
            Additional matplotlib styling options.
        """
        origin = self.node.coords
        force = self.load_pattern

        if self.node.ndof < 2:
            raise ValueError("Load plotting requires at least 2 DOFs (x, y).")

        ax.quiver(
            origin[0], origin[1],
            force[0] * scale, force[1] * scale,
            angles='xy', scale_units='xy', scale=1,
            color=kwargs.get('color', 'red'),
            width=0.015,
            headwidth=5,
            label=f"Load @ Node {self.node.id}"
        )
