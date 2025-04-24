# apeFEM/core/config.py
class FEMConfig:
    """
    Configuration class for Finite Element Method (FEM) analysis.
    This class manages the dimensional settings and degrees of freedom for FEM calculations.
    It provides class-level configuration that can be accessed throughout the FEM analysis.
    Attributes:
        dimension (int): The spatial dimension of the analysis (2 for 2D, 3 for 3D).
        nDof (int): Number of degrees of freedom per node. Default is 3 for 2D frame (UX, UY, RZ).
    Methods:
        set_dimension(dimension: int, nDof: int): Class method to set the dimension and degrees of freedom.
        Sets the spatial dimension and number of degrees of freedom for the FEM analysis.
            dimension (int): The spatial dimension (2 for 2D, 3 for 3D).
            nDof (int): Number of degrees of freedom per node.
            ValueError: If dimension or nDof are not integers.
    """
    
    @classmethod
    def set_dimension(cls, dimension:int, nDof:int):
        """_summary_

        Args:
            dimension (int): _description_
            nDof (int): _description_

        Raises:
            ValueError: _description_
        """
        
        if not isinstance(dimension, int) or not isinstance(nDof, int):
            raise ValueError("dimension and nDof must be integers")
        
        cls.dimension = dimension
        cls.nDof = nDof

