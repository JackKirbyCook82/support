# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Process Objects
@author: Jack Kirby Cook

"""

import logging
import os.path
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
__all__ = ["Scheduler", "Directory", "Calculator", "Downloader", "Saver", "Loader", "Parser", "Filter", "Filtering"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Process(ABC):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, **kwargs): pass

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Scheduler(Process, ABC):
    def __init__(self, *args, fields=[], **kwargs):
        assert isinstance(fields, list)
        self.__fields = fields

    def schedule(self, *args, **kwargs):
        assert all([field in kwargs.keys() for field in self.fields])
        contents = ODict([(field, kwargs[field]) for field in self.fields])
        contents = [[(key, value) for value in values] for key, values in contents.items()]
        for content in product(*contents):
            yield ODict(content)

    @property
    def fields(self): return self.__fields


class Breaker(object): pass
class Periodic(Process, ABC):
    def __init__(self, *args, breaker, frequency=60, **kwargs):
        super().__init__(*args, **kwargs)
        self.__frequency = frequency
        self.__breaker = breaker

    @property
    def frequency(self): return self.__frequency
    @property
    def breaker(self): return self.__breaker


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


class Directory(Process, ABC):
    def __init__(self, *args, file, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__file = file

    @property
    def directory(self): return self.file.directory
    @property
    def file(self): return self.__file


class Saver(Process, ABC):
    def __init__(self, *args, file, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__file = file

    def write(self, content, *args, folder, file, mode, **kwargs):
        file = os.path.join(folder, file) if folder is not None else file
        self.save(content, file=file, mode=mode)

    @property
    def file(self): return self.__file


class Loader(Process, ABC):
    def __init__(self, *args, file, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__file = file

    def read(self, *args, folder, file, **kwargs):
        file = os.path.join(folder, file) if folder is not None else file
        content = self.load(file=file)
        return content

    @property
    def file(self): return self.__file


class Criteria(ntuple("Criteria", "variable threshold"), ABC):
    def __call__(self, content): return self.execute(content) if bool(self) else content
    def __bool__(self): return self.threshold is not None
    def __hash__(self): return hash(self.variable)

    @abstractmethod
    def execute(self, content): pass

class Floor(Criteria):
    def execute(self, content): return content[self.variable] >= self.threshold

class Ceiling(Criteria):
    def execute(self, content): return content[self.variable] <= self.threshold


class Parser(Process, ABC):
    @staticmethod
    def unflatten(dataframe, *args, index, columns, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = [content for content in index if content in dataframe.columns]
        dataframe = dataframe.set_index(index, drop=True, inplace=False)
        columns = [value for value in columns if value in dataframe.columns]
        dataframe = dataframe[columns]
        dataset = xr.Dataset.from_dataframe(dataframe)
        return dataset

    @staticmethod
    def flatten(dataset, *args, header, **kwargs):
        assert isinstance(dataset, xr.Dataset)
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        header = [content for content in header if content in dataframe.columns]
        dataframe = dataframe[header]
        return dataframe

    @staticmethod
    def pivot(dataframe, *args, columns=[], values=[], delimiter=None, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        columns = [column for column in columns if column in dataframe.columns]
        values = [value for value in values if value in dataframe.columns]
        index = [column for column in dataframe.columns if column not in columns + values]
        dataframe = dataframe.pivot(index=index, columns=columns, values=values)
        if delimiter is not None:
            dataframe.columns = dataframe.columns.map(str(delimiter).join).str.strip(delimiter)
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @staticmethod
    def melt(dataframe, *args, name, variable, columns=[], **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = [column for column in dataframe.columns if column not in columns]
        dataframe = dataframe.melt(var_name=name, id_vars=index, value_name=variable, value_vars=columns)
        dataframe = dataframe.dropna(how="all", inplace=False)
        return dataframe

    @staticmethod
    def clean(dataframe, *args, index=[], columns, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = [index for index in index if index in dataframe.columns]
        dataframe = dataframe.drop_duplicates(subset=index, keep="last", inplace=False)
        columns = [column for column in columns if column in dataframe.columns]
        dataframe = dataframe.dropna(subset=columns, how="all", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class Filtering(object):
    FLOOR = Floor
    CEILING = Ceiling

class Filter(Process, ABC):
    def __init__(self, *args, filtering={}, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        assert isinstance(filtering, dict)
        assert all([issubclass(criteria, Criteria) for criteria in filtering.keys()])
        self.__filtering = filtering

    @typedispatcher
    def size(self, contents, *args, **kwargs): raise TypeError(type(contents).__name__)
    @typedispatcher
    def mask(self, contents, *args, **kwargs): raise TypeError(type(contents).__name__)
    @typedispatcher
    def filter(self, contents, *args, **kwargs): raise TypeError(type(contents).__name__)

    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.Series)
    def size_dataframe(self, dataframe): return len(dataframe.dropna(how="all", inplace=False).index)

    @mask.register(xr.Dataset)
    def mask_dataset(self, contents, *args, **kwargs):
        criterion = [(variable, criteria) for criteria, variables in self.filtering.items() for variable in variables if variable in contents.data_vars.keys()]
        criterion = [criteria(variable, kwargs.get(variable, None)) for (variable, criteria) in criterion]
        mask = [criteria(contents) for criteria in criterion]
        return reduce(lambda x, y: x & y, mask) if bool(mask) else None

    @mask.register(pd.DataFrame)
    def mask_dataframe(self, contents, *args, **kwargs):
        criterion = [(variable, criteria) for criteria, variables in self.filtering.items() for variable in variables if variable in contents.columns]
        criterion = [criteria(variable, kwargs.get(variable, None)) for (variable, criteria) in criterion]
        mask = [criteria(contents) for criteria in criterion]
        return reduce(lambda x, y: x & y, mask) if bool(mask) else None

    @filter.register(xr.Dataset)
    def filter_dataset(self, dataset, *args, mask, **kwargs): return dataset.where(mask, drop=True) if bool(mask is not None) else dataset
    @filter.register(pd.DataFrame)
    def filter_dataframe(self, dataframe, *args, mask, **kwargs): return dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe

    @property
    def filtering(self): return self.__filtering




