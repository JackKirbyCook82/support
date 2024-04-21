# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Process Objects
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
__all__ = ["Process", "Calculator", "Downloader", "Reader", "Writer", "Filter", "Criterion"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Process(ABC):
    @typedispatcher
    def size(self, content): raise TypeError(type(content).__name__)
    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.Series)
    def size_series(self, series): return len(series.dropna(how="all", inplace=False).index)

    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)


class Calculator(Process, ABC):
    def __init_subclass__(cls, *args, calculations={}, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        calculations = {key: value for key, value in calculations.items()}
        variable = {type(key) for key in calculations.keys()}
        assert all([callable(calculation) for calculation in calculations.values()])
        assert 0 <= len(variable) <= 1
        variable = list(variable)[0] if bool(variable) else None
        cls.__calculations__ = calculations
        cls.__variable__ = variable

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = self.__class__.__calculations__
        variable = self.__class__.__variable__
        assert variable is not None
        parameters = {field: kwargs[field] for field in variable.fields() if field in kwargs.keys()}
        for field, value in parameters.items():
            calculations = {variable: calculation for variable, calculation in calculations.items() if getattr(variable, field, None) == value}
        calculations = {variable: calculation(*args, **kwargs) for variable, calculation in calculations.items()}
        self.__calculations = calculations

    @property
    def calculations(self): return self.__calculations


class Downloader(Process, ABC):
    def __init_subclass__(cls, *args, pages={}, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__pages__ = {key: value for key, value in pages.items()}

    def __init__(self, *args, feed, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        pages = self.__class__.__pages__
        pages = {key: page(*args, feed=feed, **kwargs) for key, page in pages.items()}
        self.__pages = pages

    @property
    def pages(self): return self.__pages


class Reader(Process, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    @abstractmethod
    def read(self, *args, **kwargs): pass
    @property
    def source(self): return self.__source


class Writer(Process, ABC):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.__destination = destination

    @abstractmethod
    def write(self, query, *args, **kwargs): pass
    @property
    def destination(self): return self.__destination


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

class Filter(Process, ABC):
    def __init__(self, *args, criterion={},  **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(criterion, dict)
        assert all([issubclass(criteria, Criteria) for criteria in criterion.keys()])
        assert all([isinstance(parameter, (list, dict)) for parameter in criterion.values()])
        criterion = {criteria: parameters if isinstance(parameters, dict) else dict.fromkeys(parameters) for criteria, parameters in criterion.items()}
        criterion = [criteria(variable, threshold) for criteria, parameters in criterion.items() for variable, threshold in parameters.items()]
        self.__criterion = criterion

    @typedispatcher
    def where(self, content, mask=None): raise TypeError(type(content).__name__)
    @where.register(xr.Dataset)
    def where_dataset(self, dataset, mask=None): return dataset.where(mask, drop=True) if bool(mask is not None) else dataset
    @where.register(pd.DataFrame)
    def where_dataframe(self, dataframe, mask=None): return dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe

    def mask(self, content, variable=None):
        criterion = [criteria(content, variable=variable) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        return mask

    @property
    def criterion(self): return self.__criterion



