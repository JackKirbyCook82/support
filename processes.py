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
    def size(self, content, *args, **kwargs): raise TypeError(type(content).__name__)
    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray, *args, **kwargs): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.Series)
    def size_dataframe(self, dataframe, *args, **kwargs): return len(dataframe.dropna(how="all", inplace=False).index)


class Calculator(Process, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        calculations = {key: value for key, value in getattr(cls, "__calculations__", {}).items()}
        calculations.update(kwargs.get("calculations", {}))
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
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        pages = {key: value for key, value in getattr(cls, "__pages__", {}).items()}
        pages.update(kwargs.get("pages", {}))
        cls.__pages__ = pages

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

    @typedispatcher
    def filter(self, content, *args, **kwargs): raise TypeError(type(content).__name__)

    @filter.register(xr.Dataset)
    def dataset(self, dataset, *args, selections={}, **kwargs):
        if not bool(selections):
            mask = [criteria(dataset) for criteria in self.filtering]
            mask = reduce(lambda x, y: x & y, mask) if bool(mask) else None
            dataset = dataset.where(mask, drop=True) if bool(mask is not None) else dataset
        else:
            dataset = dataset.sel({key: value for key, value in selections.items()}).expand_dims(list(selections.keys()))
            mask = [criteria(dataset) for criteria in self.filtering]
            mask = reduce(lambda x, y: x & y, mask) if bool(mask) else None
            dataset = dataset.where(mask, drop=True) if bool(mask is not None) else dataset
            index = dataset.coords

#            broadcasting = {key: list(dataset.coords[key].values) for key in selections.keys()}
#            generator = chain(*[[(key, value) for key, value in values] for key, values in broadcasting.items()])
#            index = dataset.coords.to_dataset().drop_vars(list(broadcasting.keys()))

#            print(dataset.sel(index))
#            print(broadcasting)
#            print(dataset)
#            print(index)

        return dataset

    @filter.register(pd.DataFrame)
    def dataframe(self, dataframe, *args, selections={}, **kwargs):
        if not bool(selections):
            mask = [criteria(dataframe) for criteria in self.filtering]
            mask = reduce(lambda x, y: x & y, mask) if bool(mask) else None
            dataframe = dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe
        else:
            broadcasting = {key: list(set(dataframe.index.get_level_values(key))) for key in selections.keys()}
            generator = product(*[[(key, value) for value in values] for key, values in broadcasting.items()])

            dataframe = dataframe.iloc[reduce(lambda x, y: x & y, [dataframe.index.get_level_values(key) == value for key, value in selections.items()])]

            mask = [criteria(dataframe) for criteria in self.filtering]
            mask = reduce(lambda x, y: x & y, mask) if bool(mask) else None

            dataframe = dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe
            index = dataframe.index.to_frame().reset_index(drop=True, inplace=False).drop(list(broadcasting.keys()), axis=1, inplace=False)
            columns = [ODict(list(contents)) for contents in generator]

            print(index)
            print(columns)

        return dataframe

#            columns = [pd.DataFrame(columns, index=index) for columns in generator]
#            index = reduce(lambda indx, key: indx.droplevel(key), list(broadcasting.keys()), dataframe.index)
#            index = dataframe.index.to_frame().reset_index(drop=True, inplace=False).drop(list(broadcasting.keys()), axis=1, inplace=False)
#            index = reduce(lambda indx, key: indx.reset_index(drop=True, inplace=False).drop(key), list(selections.keys()), dataframe.index.to_frame())
#            print(index)
#            indexes = [pd.Series([value]*len(index), name=key, index=index) for (key, value) in generator]
#            index = reduce(lambda indx, other: index.append(other), indexes) if bool(indexes) else index
#            index = dataframe.index.to_frame().reset_index(drop=True, inplace=False).drop(list(broadcasting.keys()), axis=1, inplace=False)
#            print(dataframe.loc[index])
#            print(broadcasting)
#            print(dataframe)

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
    def write(self, content, *args, **kwargs): pass
    @property
    def destination(self): return self.__destination


class Loader(Reader, ABC):
    def read(self, *args, folder, mode="r", **kwargs):
        return self.source.load(*args, folder=folder, mode=mode, **kwargs)

    def reader(self, *args, mode="r", **kwargs):
        for folder in self.source.directory:
            contents = self.source.load(*args, folder=folder, mode=mode, **kwargs)
            assert isinstance(contents, dict)
            if not bool(contents):
                continue
            yield folder, contents

    @property
    def loading(self): return self.__loading


class Saver(Writer, ABC):
    def write(self, contents, *args, folder, mode, **kwargs):
        assert isinstance(contents, dict)
        if not bool(contents):
            return
        self.destination.save(contents, *args, folder=folder, names=self.saving, mode=mode, **kwargs)

    @property
    def saving(self): return self.__saving

