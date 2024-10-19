# -*- coding: utf-8 -*-
"""
Created on Weds Sept 18 2024
@name:   Filtering Objects
@author: Jack Kirby Cook

"""

import pandas as pd
import xarray as xr
from functools import reduce
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

from support.mixins import Function, Emptying, Sizing, Logging
from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Criterion"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


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


class Filter(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, criterion={}, **kwargs):
        assert isinstance(criterion, dict)
        assert all([issubclass(criteria, Criteria) for criteria in criterion.keys()])
        assert all([isinstance(parameter, (list, dict)) for parameter in criterion.values()])
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        criterion = {criteria: parameters if isinstance(parameters, dict) else dict.fromkeys(parameters) for criteria, parameters in criterion.items()}
        criterion = [criteria(variable, threshold) for criteria, parameters in criterion.items() for variable, threshold in parameters.items()]
        self.__criterion = list(criterion)

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        query, content = source
        prior = self.size(content)
        content = self.filter(content, *args, **kwargs)
        if isinstance(content, pd.DataFrame):
            content = content.reset_index(drop=True, inplace=False)
        post = self.size(content)
        string = f"Filtered: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]"
        self.logger.info(string)
        if self.empty(content): return
        return content

    def filter(self, content, *args, **kwargs):
        mask = self.mask(content)
        content = self.where(content, mask=mask)
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



