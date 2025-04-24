from dataclasses import dataclass

@dataclass
class PlotConfig:
    # Nodes
    node_color: str = "k"
    node_size: int = 10
    node_marker: str = "o"
    show_node_ids: bool = True

    # Lines
    line_color: str = "black"
    line_width: float = 1.0
    show_line_ids: bool = False       # if you later want to label lines
    line_style: str = "-"             # e.g. "-", "--", ":"

    # Grid & figure
    grid: bool = True
    aspect_equal: bool = True

    # Titles & fonts
    title_fontsize: int = 12
    label_fontsize: int = 10
