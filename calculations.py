# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation", "Calculation", "Calculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Equation(object):
    pass


class Calculation(ABC):
    pass


class Calculator(ABC):
    def __init_subclass__(cls, *args, calculations={}, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.Calculations = {key: value for key, value in calculations.items()}

    def __init__(self, *args, feed, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculations = {key: calculation(*args, feed=feed, **kwargs) for key, calculation in self.Calculations.items()}

    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def calculations(self): return self.__calculations

