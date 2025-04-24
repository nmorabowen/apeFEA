# apeFEM/core/numbering.py

class Numberer:
    def __init__(self, nodes_array, type='Plain'):
        self.type=type
        self.nodes_array=nodes_array

    def number(self):
        if self.type == 'Plain':
            self.plain()
    
    def plain(self):
        """Plain numbering method for nodes
        First come first served, no sorting or reordering.
        """
        for i,node in enumerate(self.nodes_array):
            node.id_domain = i
