from .timeseries_abstraction import TimeSeries


class ConstantTimeSeries(TimeSeries):
    """
    A time series that returns a constant factor regardless of time.

    Parameters
    ----------
    factor : float, optional
        The constant scaling factor to be applied (default is 1.0).

    Example
    -------
    >>> ts = ConstantTimeSeries(factor=2.5)
    >>> ts.get_factor(0.0)
    2.5
    """

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def get_factor(self, t: float) -> float:
        """
        Return the constant scaling factor.

        Parameters
        ----------
        t : float
            Time value (ignored in this implementation).

        Returns
        -------
        float
            Constant factor.
        """
        return self.factor


class LinearRampTimeSeries(TimeSeries):
    """
    A time series that linearly increases from 0 to 1 over the given time duration.

    Parameters
    ----------
    t_end : float, optional
        End time for the ramp (default is 1.0). For t >= t_end, the factor is 1.0.

    Example
    -------
    >>> ts = LinearRampTimeSeries(t_end=2.0)
    >>> ts.get_factor(1.0)
    0.5
    >>> ts.get_factor(3.0)
    1.0
    """

    def __init__(self, t_end: float = 1.0):
        self.t_end = t_end

    def get_factor(self, t: float) -> float:
        """
        Return the scaling factor based on linear ramp.

        Parameters
        ----------
        t : float
            Time value.

        Returns
        -------
        float
            Scaling factor (min(t / t_end, 1.0)).
        """
        return min(t / self.t_end, 1.0)
