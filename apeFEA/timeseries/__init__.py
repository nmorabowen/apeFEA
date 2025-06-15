"""
TimeSeries module — provides time-dependent scaling functions for loads or other model parameters.
"""

from .timeseries import ConstantTimeSeries, LinearRampTimeSeries

__all__ = [
    "ConstantTimeSeries",
    "LinearRampTimeSeries",
]
