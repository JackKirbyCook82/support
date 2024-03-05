# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Process Objects
@author: Jack Kirby Cook

"""

import os
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
from support.pipelines import Producer, CycleProducer, Processor, Consumer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "Calculator", "Downloader", "CycleDownloader", "Parser", "Filter", "Pivoter", "Filtering", "Axes"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Files(object):
    def __init__(self, *args, file, **kwargs):
        super().__init__(*args, **kwargs)
        self.__file = file

#    def directory(self, *args, **kwargs): return self.file.directory(*args, **kwargs)
#    def path(self, *args, **kwargs): return self.file.path(*args, **kwargs)
#    def parse(self, *args, **kwargs): return self.file.parse(*args, **kwargs)
#    def read(self, *args, **kwargs): return self.file.read(*args, **kwargs)
#    def write(self, *args, **kwargs): self.file.write(*args, **kwargs)

    @property
    def file(self): return self.__file


class Saver(Files, Consumer, ABC, title="Saved"):
    pass

#    @typedispatcher
#    def write(self, content, *args, **kwargs): raise TypeError(type(content).__name__)

#    @write.register(dict)
#    def write_multiple(self, contents, *args, folder=None, **kwargs):
#        assert "file" not in kwargs.keys()
#        for file, content in contents.items():
#            self.write(content, *args, folder=folder, file=file, **kwargs)

#    @write.register(xr.Dataset, pd.DataFrame)
#    def write_single(self, content, *args, folder=None, file, filemode, **kwargs):
#        folder = os.path.join(self.repository, folder) if bool(folder) else self.repository
#        os.makedirs(folder)
#        file = os.path.join(folder, file)
#        self.save(content, file=file, filemode=filemode)
#        __logger__.info("Saved: {}[{}]".format(repr(self), str(file)))


class Loader(Files, Producer, ABC, title="Loaded"):
    pass

#    def read(self, *args, **kwargs):
#        pass


class Calculations(object):
    def __init_subclass__(cls, *args, **kwargs):
        calculations = getattr(cls, "__calculations__", {})
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


class Websites(object):
    def __init_subclass__(cls, *args, **kwargs):
        pages = getattr(cls, "__pages__", {})
        pages.update(kwargs.get("pages", {}))
        cls.__pages__ = pages

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        pages = self.__class__.__pages__
        pages = {key: page(*args, **kwargs) for key, page in pages.items()}
        self.__pages = pages

    @property
    def pages(self): return self.__pages


class Calculator(Calculations, Processor, ABC, title="Calculated"): pass
class Downloader(Websites, Producer, ABC, title="Downloaded"): pass
class CycleDownloader(Websites, CycleProducer, ABC, title="Downloaded"): pass


class Axes(ntuple("Axes", "index columns values scope")):
    def __new__(cls, index, columns, values, scope):
        assert isinstance(index, (str, dict))
        assert len(index) == 1 if isinstance(index, dict) else True
        assert all([isinstance(contents, (list, dict)) for contents, in (columns, values, scope)])
        index = list(index.keys()) if isinstance(index, dict) else str(index)
        columns = list(columns.keys()) if isinstance(columns, dict) else list(columns)
        values = list(values.keys()) if isinstance(values, dict) else list(values)
        scope = list(scope.keys()) if isinstance(scope, dict) else list(scope)
        return super().__new__(cls, index, columns, values, scope)

    @property
    def header(self): return [self.index] + self.columns + self.values + self.scope


class Table(Processor, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.axes = kwargs.get("axes", getattr(cls, "axes", None))

    @property
    def header(self): return type(self).axes.header
    @property
    def index(self): return type(self).axes.index
    @property
    def columns(self): return type(self).axes.columns
    @property
    def values(self): return type(self).axes.values
    @property
    def scope(self): return type(self).axes.scope


class Pivoter(Table, ABC, title="Pivoted"):
    def __init__(self, *args, delimiter="|", **kwargs):
        super().__init__(*args, **kwargs)
        self.__delimiter = delimiter

    @typedispatcher
    def pivot(self, contents, *args, **kwargs): raise TypeError(type(contents).__name__)

    @pivot.register(xr.Dataset)
    def pivot_dataset(self, dataset, *args, **kwargs):
        assert self.index in dataset.coords.keys()
        assert all([column in dataset.data_vars.keys() for column in self.columns])
        scope = [content for content in dataset.coords.keys() if content in self.scope]
        values = [content for content in dataset.data_vars.keys() if content in self.values]
        assert all([len(dataset[content].values) == 0 for content in dataset.coords.keys() if content in scope])
        string = lambda column: str(self.delimiter).join([content.value for content in column])
        dataset = dataset[values]
        columns = [[(content, value) for value in list(dataset[content].values)] for content in self.columns]
        columns = [ODict(list(contents)) for contents in product(*columns)]
        for content in scope:
            dataset.squeeze(content)
        dataset = [dataset.sel(column).rename({self.index: string(column)}) for column in columns]
        return dataset

    @pivot.register(pd.DataFrame)
    def pivot_dataframe(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe.index, pd.RangeIndex)
        assert self.index in dataframe.columns
        assert all([column in dataframe.columns for column in self.columns])
        scope = [content for content in dataframe.columns if content in self.scope]
        values = [content for content in dataframe.columns if content in self.values]
        assert all([len(set(dataframe[content].values)) == 0 for content in dataframe.columns if content in scope])
        scope = {content: list(dataframe[content].values)[0] for content in self.scope}
        dataframe = dataframe[[self.index] + self.columns + values]
        dataframe = dataframe.pivot(index=self.index, columns=self.columns, values=values)
        dataframe.columns = dataframe.columns.map(str(self.delimiter).join).str.strip(self.delimiter)
        for key, value in scope.items():
            dataframe[key] = value
        return dataframe

    @property
    def delimiter(self): return self.__delimiter


class Parser(Table, ABC, title="Parsed"):
    @typedispatcher
    def parse(self, contents, *args, **kwargs): raise TypeError(type(contents).__name__)
    @typedispatcher
    def prune(self, contents, *args, **kwargs): raise TypeError(type(contents).__name__)

    @parse.register(xr.Dataset)
    def parse_dataset(self, dataset, *args, **kwargs):
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        header = [content for content in self.header if content in dataframe.columns]
        dataframe = dataframe[header]
        return dataframe

    @parse.register(pd.DataFrame)
    def parse_dataframe(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe.index, pd.RangeIndex)
        index = [self.index] + self.columns + self.scope
        index = [content for content in index if content in dataframe.columns]
        dataframe = dataframe.set_index(index, drop=True, inplace=False)
        columns = [value for value in self.values if value in dataframe.columns]
        dataframe = dataframe[columns]
        dataset = xr.Dataset.from_dataframe(dataframe)
        return dataset

    @prune.register(pd.DataFrame)
    def prune_dataframe(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe.index, pd.RangeIndex)
        index = [index for index in dataframe.columns if index in self.index]
        dataframe = dataframe.drop_duplicates(subset=index, keep="last", inplace=False)
        columns = [column for column in dataframe.columns if column in self.columns]
        dataframe = dataframe.dropna(subset=columns, how="all", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @property
    def parsing(self): return self.__parsing


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


class Filtering(object):
    FLOOR = Floor
    CEILING = Ceiling

class Filter(Processor, ABC, title="Filtered"):
    def __init__(self, *args, filtering={}, **kwargs):
        super().__init__(*args, **kwargs)
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



