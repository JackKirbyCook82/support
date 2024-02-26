# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Process Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd
import xarray as xr
from abc import ABC
from enum import IntEnum
from functools import reduce
from collections import namedtuple as ntuple

from support.pipelines import Producer, Processor, Consumer
from support.dispatchers import kwargsdispatcher, typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "Calculator", "Parser", "Filter", "Parsing", "Filtering"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


Parsing = IntEnum("Parsing", ["FLATTEN", "UNFLATTEN"], start=1)
Filtering = IntEnum("Filtering", ["FLOOR", "CEILING"], start=1)


class Files(object):
    def __init__(self, *args, file, **kwargs):
        super().__init__(*args, **kwargs)
        self.__file = file

    def directory(self, *args, **kwargs): return self.file.directory(*args, **kwargs)
    def path(self, *args, **kwargs): return self.file.path(*args, **kwargs)
    def parse(self, *args, **kwargs): return self.file.parse(*args, **kwargs)
    def read(self, *args, **kwargs): return self.file.read(*args, **kwargs)
    def write(self, *args, **kwargs): self.file.write(*args, **kwargs)

    @property
    def file(self): return self.__file


class Saver(Files, Consumer, ABC, title="Saved"): pass
class Loader(Files, Producer, ABC, title="Loaded"): pass


class Calculator(Processor, ABC, title="Calculated"):
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


class Formatting(object):
    def __init_subclass__(cls, *args, index=[], columns=[], **kwargs):
        assert isinstance(index, (dict, list)) and isinstance(columns, (dict, list))
        Axes = ntuple("Axes", "index columns")
        index = list(index.keys() if isinstance(index, dict) else index)
        columns = list(columns.keys() if isinstance(columns, dict) else columns)
        cls.__axes__ = Axes(index, columns)

    def __init__(self, *args, drop=False, **kwargs):
        assert isinstance(drop, bool)
        super().__init__(*args, **kwargs)
        self.__axes = self.__class__.__axes__
        self.__drop = drop

    @typedispatcher
    def format(self, content, *args, **kwargs): raise TypeError(type(content).__name__)
    @format.register(pd.DataFrame)
    def dataframe(self, dataframe, *args, **kwargs):
        index = [index for index in dataframe.columns if index in self.index]
        dataframe = dataframe.drop_duplicates(subset=index, keep="last", inplace=False)
        columns = [column for column in dataframe.columns if column in self.columns]
        dataframe = dataframe.dropna(subset=columns, how="all", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @property
    def header(self): return self.index + self.columns
    @property
    def columns(self): return self.axes.columns
    @property
    def index(self): return self.axes.index

    @property
    def axes(self): return self.__axes
    @property
    def drop(self): return self.__drop


class Parser(Formatting, Processor, ABC, title="Parsed"):
    def __init__(self, *args, parsing, **kwargs):
        assert parsing in Parsing
        super().__init__(*args, **kwargs)
        self.__parsing = parsing

    def parse(self, contents, *args, **kwargs):
        transform = (isinstance(contents, pd.DataFrame) and self.parsing is Parsing.UNFLATTEN) or (isinstance(contents, xr.Dataset) and self.parsing is Parsing.FLATTEN)
        formatting = bool(self.drop) and ((isinstance(contents, pd.DataFrame) and not bool(transform)) or (isinstance(contents, xr.Dataset) and bool(transform)))
        contents = self.transform(contents, *args, parsing=self.parsing, **kwargs) if bool(transform) else contents
        contents = self.format(contents, *args, **kwargs) if bool(formatting) else contents
        return contents

    @kwargsdispatcher("parsing")
    def transform(self, content, *args, parsing, **kwargs): raise ValueError(str(parsing.name).title())

    @transform.register.value(Parsing.FLATTEN)
    def flatten(self, dataset, *args, **kwargs):
        assert isinstance(dataset, xr.Dataset)
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        header = [value for value in self.header if value in dataframe.columns]
        dataframe = dataframe[header]
        return dataframe

    @transform.register.value(Parsing.UNFLATTEN)
    def unflatten(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = [value for value in self.index if value in dataframe.columns]
        dataframe = dataframe.set_index(index, drop=True, inplace=False)
        columns = [value for value in self.columns if value in dataframe.columns]
        dataframe = dataframe[columns]
        dataset = xr.Dataset.from_dataframe(dataframe)
        return dataset

    @property
    def parsing(self): return self.__parsing


class Filter(Formatting, Processor, ABC, title="Filtered"):
    def __init__(self, *args, filtering={}, **kwargs):
        assert isinstance(filtering, dict)
        super().__init__(*args, **kwargs)
        self.__filtering = filtering

    def mask(self, content, *args, **kwargs):
        mask = {key: {variable: kwargs.get(variable, None) for variable in variables} for key, variables in self.filtering.items()}
        mask = {key: {variable: threshold for variable, threshold in variables.items() if threshold is not None} for key, variables in mask.items()}
        mask = [dict(filtering=filtering, variable=variable, threshold=threshold) for filtering, variables in mask.items() for variable, threshold in variables.items()]
        mask = [self.criteria(content, *args, **parameters, **kwargs) for parameters in mask]
        return reduce(lambda x, y: x & y, mask) if bool(mask) else None

    def filter(self, contents, *args, mask, **kwargs):
        filtering, formatting = bool(mask is not None), bool(self.drop)
        contents = self.filtration(contents, *args, mask=mask, **kwargs) if bool(filtering) else contents
        contents = self.format(contents, *args, mask=mask, **kwargs) if bool(formatting) else contents
        return contents

    @kwargsdispatcher("filtering")
    def criteria(self, content, *args, filtering, **kwargs): raise ValueError(str(filtering.name).title())
    @criteria.register.value(Filtering.FLOOR)
    def floor(self, content, *args, variable, threshold, **kwargs): return content[variable] >= threshold
    @criteria.register.value(Filtering.CEILING)
    def ceiling(self, content, *args, variable, threshold, **kwargs): return content[variable] <= threshold

    @typedispatcher
    def filtration(self, content, *args, **kwargs): raise TypeError(type(content).__name__)
    @filtration.register(xr.Dataset)
    def dataset(self, dataset, *args, mask, **kwargs): return dataset.where(mask, drop=True)
    @filtration.register(pd.DataFrame)
    def dataframe(self, dataframe, *args, mask, **kwargs): return dataframe.where(mask).dropna(how="all", inplace=False)

    @property
    def filtering(self): return self.__filtering




