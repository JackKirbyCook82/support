# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Filtering Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Criterion"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Criteria(ntuple("Criteria", "variable threshold"), ABC):
    def __repr__(self): return f"{type(self).__name__}[{str(self.variable)}, {str(self.threshold)}]"
    def __call__(self, content, variable):
        column = self.column(variable)
        return self.execute(content, column)

    @typedispatcher
    def column(self, variable): pass
    @column.register(type(None))
    def single(self, empty): return self.variable
    @column.register(str)
    def double(self, variable): return tuple([self.variable, variable])
    @column.register(list)
    def multiple(self, variables): return tuple([self.variable] + list(variables))

    @abstractmethod
    def execute(self, content, column): pass


class Floor(Criteria):
    def execute(self, content, column): return content[column] >= self.threshold

class Ceiling(Criteria):
    def execute(self, content, column): return content[column] <= self.threshold

class Null(Criteria):
    @typedispatcher
    def execute(self, content, column): raise TypeError(type(content).__name__)
    @execute.register(pd.DataFrame)
    def dataframe(self, content, column): return content[column].notna()
    @execute.register(xr.Dataset)
    def dataset(self, content, column): return content[column].notnull()


class Criterion(object):
    FLOOR = Floor
    CEILING = Ceiling
    NULL = Null


class Filter(ABC):
    def __init_subclass__(cls, *args, variable, **kwargs):
        cls.__variable__ = variable

    def __init__(self, *args, criterion={},  **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(criterion, dict)
        assert all([issubclass(criteria, Criteria) for criteria in criterion.keys()])
        assert all([isinstance(parameter, (list, dict)) for parameter in criterion.values()])
        criterion = {criteria: parameters if isinstance(parameters, dict) else dict.fromkeys(parameters) for criteria, parameters in criterion.items()}
        criterion = [criteria(variable, threshold) for criteria, parameters in criterion.items() for variable, threshold in parameters.items()]
        self.__variable = self.__class__.__variable__
        self.__criterion = criterion

    def mask(self, content, variable=None):
        criterion = [criteria(content, variable=variable) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        return mask

    @typedispatcher
    def where(self, content, mask=None): raise TypeError(type(content).__name__)
    @where.register(xr.Dataset)
    def where_dataset(self, dataset, mask=None): return dataset.where(mask, drop=True) if bool(mask is not None) else dataset
    @where.register(pd.DataFrame)
    def where_dataframe(self, dataframe, mask=None): return dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe

    @typedispatcher
    def size(self, content): raise TypeError(type(content).__name__)
    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.DataFrame)
    def size_dataframe(self, dataframe): return len(dataframe.dropna(how="all", inplace=False).index)
    @size.register(pd.Series)
    def size_series(self, series): return len(series.dropna(how="all", inplace=False).index)

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def variable(self): return self.__variable
    @property
    def criterion(self): return self.__criterion

