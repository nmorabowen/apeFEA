# apeFEA/geometry/plot_geometry.py

import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class PlotGeometry:
    """Mixin class with ploting capabilities for the Geometry class.
    """

    def plot_nodes(self, backend: str = "matplotlib", show_ids: bool = True, ax=None, show: bool = True):
        """
        Plot nodes using either matplotlib or plotly, automatically choosing 2D or 3D.

        Args:
            backend (str): 'matplotlib' or 'plotly'.
            show_ids (bool): Whether to display node IDs.
            ax (matplotlib axis or None): Optional axis to plot on (only for matplotlib).
            show (bool): Whether to call plt.show() or fig.show().

        Returns:
            tuple or Figure: (fig, ax) for matplotlib, or fig for plotly.
        """
        
        if not hasattr(self, 'nodes') or not hasattr(self.nodes, 'nodes_array') or not hasattr(self, 'name'):
            raise AttributeError("Class using PlottingMixin must define 'nodes' with 'nodes_array' and a 'name'.")


        if not self.nodes.nodes_array:
            logger.warning("No nodes to plot.")
            return

        dim = len(self.nodes.nodes_array[0].coords)
        coords = list(zip(*[node.coords for node in self.nodes.nodes_array]))
        ids = [str(node.id) for node in self.nodes.nodes_array]

        if backend == "matplotlib":
            if dim == 2:
                return self._plot_2d_matplotlib(coords, ids, show_ids, ax, show)
            elif dim == 3:
                return self._plot_3d_matplotlib(coords, ids, show_ids, ax, show)
        elif backend == "plotly":
            if dim == 2:
                return self._plot_2d_plotly(coords, ids, show_ids, show)
            elif dim == 3:
                return self._plot_3d_plotly(coords, ids, show_ids, show)
        raise ValueError("Only 2D or 3D plotting is supported for backends 'matplotlib' or 'plotly'.")

    def _plot_2d_matplotlib(self, coords, ids, show_ids, ax=None, show=True):
        x, y = coords
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            fig = ax.figure

        ax.scatter(x, y, color='blue', label='Nodes')
        if show_ids:
            for i, txt in enumerate(ids):
                ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), ha='center')
        ax.set_aspect('equal')
        ax.set_title(f"Node Plot - Geometry: {self.name}")
        ax.grid(True)

        if created_fig and show:
            plt.show()

        return fig, ax

    def _plot_3d_matplotlib(self, coords, ids, show_ids, ax=None, show=True):
        x, y, z = coords
        created_fig = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            fig = ax.figure

        ax.scatter(x, y, z, color='blue')
        if show_ids:
            for i, txt in enumerate(ids):
                ax.text(x[i], y[i], z[i], txt)
        ax.set_title(f"3D Node Plot - Geometry: {self.name}")

        if created_fig and show:
            plt.show()

        return fig, ax

    def _plot_2d_plotly(self, coords, ids, show_ids, show=True):
        x, y = coords
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text' if show_ids else 'markers',
            text=ids if show_ids else None,
            textposition="top center",
            marker=dict(size=8, color='blue')
        ))
        fig.update_layout(
            title=f"Node Plot - Geometry: {self.name}",
            xaxis_title="X", yaxis_title="Y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=False
        )

        if show:
            fig.show()
        return fig

    def _plot_3d_plotly(self, coords, ids, show_ids, show=True):
        x, y, z = coords
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text' if show_ids else 'markers',
            text=ids if show_ids else None,
            marker=dict(size=4, color='blue')
        ))
        fig.update_layout(
            title=f"3D Node Plot - Geometry: {self.name}",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode='data'
            ),
            showlegend=False
        )

        if show:
            fig.show()
        return fig
    
    def plot_lines(self,
                   backend: str = "matplotlib",
                   ax=None,
                   show: bool = True,
                   line_kwargs: dict = None):
        """
        Plot 2D line elements by connecting each line's start/end node coords.
        """
        if not hasattr(self, "get_lines_list"):
            raise AttributeError("Need `get_lines_list()` returning your line objects.")
        lines = self.get_lines_list()
        if not lines:
            logger.warning("No lines to plot.")
            return

        line_coords = [(ln.node_start.coords, ln.node_end.coords) for ln in lines]

        if backend == "matplotlib":
            return self._plot_2d_lines_matplotlib(line_coords, ax, show, line_kwargs)
        elif backend == "plotly":
            return self._plot_2d_lines_plotly(line_coords, show, line_kwargs)
        else:
            raise ValueError("Only backends 'matplotlib' or 'plotly' supported.")

    def _plot_2d_lines_matplotlib(self,
                                  line_coords,
                                  ax=None,
                                  show: bool = True,
                                  line_kwargs: dict = None):
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            fig = ax.figure

        kws = {"color": "black", "linewidth": 1.5}
        if line_kwargs:
            kws.update(line_kwargs)

        for p0, p1 in line_coords:
            x = [p0[0], p1[0]]
            y = [p0[1], p1[1]]
            ax.plot(x, y, **kws)

        ax.set_aspect("equal")
        ax.set_title(f"Line Plot - Geometry: {self.name}")
        ax.grid(True)

        if created_fig and show:
            plt.show()

        return fig, ax

    def _plot_2d_lines_plotly(self,
                              line_coords,
                              show: bool = True,
                              line_kwargs: dict = None):
        fig = go.Figure()
        default_line = {"color": "black", "width": 2}

        for p0, p1 in line_coords:
            fig.add_trace(go.Scatter(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                mode="lines",
                line=line_kwargs or default_line
            ))

        fig.update_layout(
            title=f"Line Plot - Geometry: {self.name}",
            xaxis_title="X", yaxis_title="Y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=False
        )

        if show:
            fig.show()

        return fig
    
    
    