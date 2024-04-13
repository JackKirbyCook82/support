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
from itertools import product
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculator", "Downloader", "Reader", "Writer", "Loader", "Saver", "Filter", "Filtering"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Process(ABC):
    @typedispatcher
    def size(self, content): raise TypeError(type(content).__name__)
    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.Series)
    def size_dataframe(self, dataframe): return len(dataframe.dropna(how="all", inplace=False).index)


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


class Criteria(ntuple("Criteria", "variable threshold"), ABC):
    def __repr__(self): return f"{type(self).__name__}[{str(self.variable)}, {str(self.threshold)}]"
    def __call__(self, content): return self.execute(content)

    @abstractmethod
    def execute(self, content): pass

class Floor(Criteria):
    def execute(self, content): return content[self.variable] >= self.threshold

class Ceiling(Criteria):
    def execute(self, content): return content[self.variable] <= self.threshold

class Null(Criteria):
    @typedispatcher
    def execute(self, content): raise TypeError(type(content).__name__)
    @execute.register(pd.DataFrame)
    def dataframe(self, content): return content[self.variable].notna()
    @execute.register(xr.Dataset)
    def dataset(self, content): return content[self.variable].notnull()


class Filtering(object):
    FLOOR = Floor
    CEILING = Ceiling
    NULL = Null

class Filter(Process, ABC):
    def __init__(self, *args, filtering={},  **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(filtering, dict)
        assert all([issubclass(criteria, Criteria) for criteria in filtering.keys()])
        assert all([isinstance(parameter, (list, dict)) for parameter in filtering.values()])
        filtering = {criteria: parameters if isinstance(parameters, dict) else dict.fromkeys(parameters) for criteria, parameters in filtering.items()}
        filtering = [criteria(variable, threshold) for criteria, parameters in filtering.items() for variable, threshold in parameters.items()]
        self.__filtering = filtering

    def filter(self, content, *args, select={}, **kwargs):
        if not bool(select):
            criterion = self.criterion(content, self.filtering)
            mask = self.mask(criterion)
            content = self.where(content, mask)
        else:
            axes = self.axes(content, select)
            selected = self.select(content, select)
            criterion = self.criterion(selected, self.filtering)
            mask = self.mask(criterion)
            selected = self.where(selected, mask)
            index = self.index(selected, axes)
            content = content.loc[index]
        return content

    @typedispatcher
    def where(self, content, mask=None): raise TypeError(type(content).__name__)
    @where.register(xr.Dataset)
    def where_dataset(self, dataset, mask=None): return dataset.where(mask, drop=True) if bool(mask is not None) else dataset
    @where.register(pd.DataFrame)
    def where_dataframe(self, dataframe, mask=None): return dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe

    @typedispatcher
    def select(self, content, select={}): raise TypeError(type(content).__name__)
    @select.register(xr.Dataset)
    def select_dataset(self, dataset, select={}): return dataset.sel({key: value for key, value in select.items()}).expand_dims(list(select.keys()))
    @select.register(pd.DataFrame)
    def select_dataframe(self, dataframe, select={}): return dataframe.iloc[reduce(lambda x, y: x & y, [dataframe.index.get_level_values(key) == value for key, value in select.items()])]

    @typedispatcher
    def axes(self, content, select={}): raise TypeError(type(content).__name__)
    @axes.register(xr.Dataset)
    def axes_dataset(self, dataset, select={}): return ODict([(key, list(dataset.coords[key].values)) for key in select.keys()])
    @axes.register(pd.DataFrame)
    def axes_dataframe(self, dataframe, select={}): return ODict([(key, list(set(dataframe.index.get_level_values(key)))) for key in select.keys()])

    @typedispatcher
    def index(self, content, axes={}): raise TypeError(type(content).__name__)
    @index.register(xr.Dataset)
    def index_dataset(self, dataset, axes={}): return ODict([(key, list(dataarray.values)) for key, dataarray in dataset.coords.items() if key not in axes.keys()]) | axes
    @index.register(pd.DataFrame)
    def index_dataframe(self, dataframe, axes={}):
        index = dataframe.index.to_frame().reset_index(drop=True, inplace=False).drop(list(axes.keys()), axis=1, inplace=False)
        axes = product(*[[(key, value) for value in values] for key, values in axes.items()])
        axes = [pd.DataFrame.from_records({key: [value] * len(index) for (key, value) in list(axis)}) for axis in iter(axes)]
        index = pd.concat([pd.concat([index, axis], axis=1) for axis in axes], axis=0)
        return pd.MultiIndex.from_frame(index[dataframe.index.names])

    @staticmethod
    def criterion(content, filtering): return [criteria(content) for criteria in filtering]
    @staticmethod
    def mask(criterion): return reduce(lambda x, y: x & y, criterion) if bool(criterion) else None

    @property
    def filtering(self): return self.__filtering


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


class Loader(Reader):
    def __init_subclass__(cls, *args, query=lambda folder: dict(), **kwargs):
        super().__init_subclass__(*args, **kwargs)
        assert callable(query)
        cls.__query__ = query

    def __init__(self, *args, mode="r", **kwargs):
        super().__init__(*args, **kwargs)
        self.__query = self.__class__.__query__
        self.__mode = mode

    def execute(self, *args, **kwargs):
        for folder in self.source.directory:
            query = self.query(folder)
            contents = self.read(*args, folder=folder, **kwargs)
            assert isinstance(query, dict) and isinstance(contents, dict)
            if not bool(contents):
                continue
            yield query | contents

    def read(self, *args, folder=None, **kwargs):
        return self.source.load(*args, folder=folder, mode=self.mode, **kwargs)

    @property
    def query(self): return self.__query
    @property
    def mode(self): return self.__mode


class Saver(Writer):
    def __init_subclass__(cls, *args, folder=lambda query: None, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        assert callable(folder)
        cls.__folder__ = folder

    def __init__(self, *args, mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.__folder = self.__class__.__folder__
        self.__mode = mode

    def execute(self, query, *args, **kwargs):
        assert isinstance(query, dict)
        folder = self.folder(query)
        self.write(query, *args, folder=folder, **kwargs)

    def write(self, query, *args, folder=None, **kwargs):
        self.destination.save(query, *args, folder=folder, mode=self.mode, **kwargs)

    @property
    def folder(self): return self.__folder
    @property
    def mode(self): return self.__mode



