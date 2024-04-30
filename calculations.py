# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, ABCMeta, abstractmethod

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Equation", "Calculation", "Calculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Variable(object):
    pass


class Equation(object):
    pass


class CalculationMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        equation = kwargs.get("equation", getattr(cls, "__equation__", None))
        fields = {key: value for key, value in getattr(cls, "__fields__", {}).items()}
        fields = fields | dict.fromkeys(kwargs.get("fields", []))
        fields = fields | {key: kwargs.get(key, None) for key in fields.items()}
        assert all([attr not in fields.keys() for attr in ("domain", "equation")])
        cls.__equation__ = equation
        cls.__fields__ = fields


class Calculation(ABC, metaclass=CalculationMeta):
    pass


class Calculator(ABC):
    def __init_subclass__(cls, *args, calculations={}, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.Calculations = {key: value for key, value in calculations.items()}

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculations = {key: calculation(*args, **kwargs) for key, calculation in self.Calculations.items()}

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

