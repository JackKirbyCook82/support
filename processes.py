# -*- coding: utf-8 -*-
"""
Created on Weds Sept 18 2024
@name:   Processes Objects
@author: Jack Kirby Cook

"""


import time
import logging
import pandas as pd
import xarray as xr
from functools import reduce
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

from support.mixins import Sizing
from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Criterion", "Calculator", "Downloader"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Process(Sizing, ABC):
    def __init_subclass__(cls, **kwargs):
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__title = self.__class__.__title__
        self.__logger = __logger__

    def __repr__(self): return f"{str(self.name)}"
    def __call__(self, *args, **kwargs):
        start = time.time()
        results = self.execute(*args, **kwargs)
        elapsed = (time.time() - start).total_seconds()
        string = f"{str(self.title).title()}: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)
        return results

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def logger(self): return self.__logger
    @property
    def title(self): return self.__title
    @property
    def name(self): return self.__name


class Criteria(ntuple("Criteria", "variable threshold"), ABC):
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
    def execute_dataframe(self, content): return content[self.variable].notna()
    @execute.register(xr.Dataset)
    def execute_dataset(self, content): return content[self.variable].notnull()

class Criterion(object):
    FLOOR = Floor
    CEILING = Ceiling
    NULL = Null


class Filter(Process, ABC, title="Filtered"):
    def __init__(self, *args, criterion={}, name=None, **kwargs):
        assert isinstance(criterion, dict)
        assert all([issubclass(criteria, Criteria) for criteria in criterion.keys()])
        assert all([isinstance(parameter, (list, dict)) for parameter in criterion.values()])
        super().__init__(*args, name=name, **kwargs)
        criterion = {criteria: parameters if isinstance(parameters, dict) else dict.fromkeys(parameters) for criteria, parameters in criterion.items()}
        criterion = [criteria(variable, threshold) for criteria, parameters in criterion.items() for variable, threshold in parameters.items()]
        self.__criterion = dict(criterion)

    def filter(self, content, *args, variable, **kwargs):
        prior = self.size(content)
        mask = self.mask(content)
        content = self.where(content, mask=mask)
        post = self.size(content)
        string = f"Filtered: {repr(self)}|{str(variable)}[{prior:.0f}|{post:.0f}]"
        self.logger.info(string)
        return content

    def mask(self, content):
        criterion = [criteria(content) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        return mask

    @typedispatcher
    def where(self, content, mask=None): raise TypeError(type(content).__name__)
    @where.register(xr.Dataset)
    def where_dataset(self, dataset, mask=None): return dataset.where(mask, drop=True) if bool(mask is not None) else dataset
    @where.register(pd.DataFrame)
    def where_dataframe(self, dataframe, mask=None): return dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe

    @property
    def criterion(self): return self.__criterion


class Calculator(Process, ABC, title="Calculated"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        calculations = getattr(cls, "__calculations__", {}) | kwargs.get("calculations", {})
        calculation = kwargs.get("calculation", getattr(cls, "__calculation__", None))
        variables = kwargs.get("variables", getattr(cls, "__variables__", None))
        cls.__calculations__ = calculations
        cls.__calculation__ = calculation
        cls.__variables__ = variables

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = self.__class__.__calculations__
        calculation = self.__class__.__calculation__
        variables = self.__class__.__variables__
        calculations = {key: calculation(*args, **kwargs) for key, calculation in calculations.items()}
        calculation = calculation(*args, **kwargs) if calculation is not None else calculation
        variables = variables(*args, **kwargs) if variables is not None else variables
        self.__calculations = calculations
        self.__calculation = calculation
        self.__variables = variables

    @property
    def calculations(self): return self.__calculations
    @property
    def calculation(self): return self.__calculation
    @property
    def variables(self): return self.__variables


class Downloader(Process, ABC, title="Downloaded"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        pages = getattr(cls, "__pages__", {}) | kwargs.get("pages", {})
        cls.__pages__ = pages

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        pages = self.__class__.__pages__
        pages = {key: page(*args, **kwargs) for key, page in pages.items()}
        self.__pages = pages

    @property
    def pages(self): return self.__pages




