# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Process Objects
@author: Jack Kirby Cook

"""

import logging
from abc import ABC
import numpy as np
import pandas as pd
import xarray as xr
from enum import IntEnum
from functools import reduce
from collections import namedtuple as ntuple

from support.pipelines import Producer, Processor, Consumer
from support.dispatchers import kwargsdispatcher, typedispatcher
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "Parser", "Filter", "Parsing", "Filtering"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


Parsing = IntEnum("Parsing", ["FLATTEN", "UNFLATTEN"], start=1)
Filtering = IntEnum("Filtering", ["LOWER", "UPPER"], start=1)


class Saver(Consumer, ABC, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = file


class Loader(Producer, ABC, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = file


class Parser(Processor, ABC, title="Parsed"):
    def __init_subclass__(cls, *args, scope=[], index=[], columns=[], **kwargs):
        super().__init_subclass__(*args, **kwargs)
        Parameters = ntuple("Parameters", "scope index columns")
        cls.parameters = Parameters(scope, index, columns)

    def __init__(self, *args, parsing, **kwargs):
        super().__init__(*args, **kwargs)
        assert parsing in Parsing
        self.__parsing = parsing

    def parse(self, contents, *args, **kwargs):
        return self.function(contents, *args, parsing=self.parsing, **kwargs)

    @kwargsdispatcher("parsing")
    def function(self, content, *args, parsing, **kwargs): raise ValueError(str(parsing.name).title())

    @function.register.value(Parsing.FLATTEN)
    def flatten(self, dataset, *args, **kwargs):
        assert isinstance(dataset, xr.Dataset)
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        columns = [column for column in self.index + self.scope if column in dataframe.columns]
        dataframe = dataframe.drop_duplicates(subset=columns, keep="last", inplace=False)
        dataframe = dataframe.set_index(columns, drop=True, inplace=False)
        columns = [column for column in self.columns if column in dataframe.columns]
        return dataframe[columns]

    @function.register.value(Parsing.UNFLATTEN)
    def unflatten(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        columns = [column for column in self.index + self.scope if column in dataframe.columns]
        dataframe = dataframe.drop_duplicates(subset=columns, keep="last", inplace=False)
        dataframe = dataframe.set_index(columns, drop=True, inplace=False)
        dataset = xr.Dataset.from_dataframe(dataframe)
        for value in self.scope:
            dataset = dataset.squeeze(value)
        columns = [column for column in self.columns if column in dataset.data_vars.keys()]
        return dataset[columns]

    @property
    def columns(self): return self.parameters.columns
    @property
    def index(self): return self.parameters.index
    @property
    def scope(self): return self.parameters.scope
    @property
    def parsing(self): return self.__parsing


class FilterFunction(ntuple("FilterFunction", "variable threshold"), ABC, metaclass=RegistryMeta):
    def __call__(self, content): return self.execute(content)

class LowerFilterFunction(FilterFunction, key=Filtering.LOWER):
    def execute(self, content): return content[self.variable] >= self.threshold

class UpperFilterFunction(FilterFunction, key=Filtering.UPPER):
    def execute(self, content): return content[self.variable] <= self.threshold


class Filter(Processor, ABC, title="Filtered"):
    def __init__(self, *args, filtering={}, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(filtering, dict)
        self.__filtering = filtering

    def filter(self, contents, *args, mask, **kwargs):
        return self.function(contents, *args, mask=mask, **kwargs) if mask is not None else contents

    def mask(self, content, *args, **kwargs):
        filtering = {key: {variable: kwargs.get(variable, None) for variable in variables} for key, variables in self.filtering.items()}
        filtering = {key: {variable: threshold for variable, threshold in variables.items() if threshold is not None} for key, variables in filtering.items()}
        filtering = [FilterFunction[key](variable, threshold) for key, variables in filtering.items() for variable, threshold in variables.items()]
        mask = [function(content) for function in filtering]
        return reduce(lambda x, y: x & y, mask) if bool(mask) else None

    @typedispatcher
    def function(self, content, *args, mask, **kwargs): raise TypeError(type(content).__name__)

    @function.register(pd.DataFrame)
    def dataframe(self, dataframe, *args, mask, **kwargs):
        dataframe = dataframe.where(mask)
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @function.register(xr.Dataset)
    def dataset(self, dataset, *args, mask, **kwargs):
        dataset = dataset.where(mask, drop=True)
        return dataset

    @property
    def filtering(self): return self.__filtering



