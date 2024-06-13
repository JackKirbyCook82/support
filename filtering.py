# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Filtering Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd
import xarray as xr
from functools import reduce
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher
from support.pipelines import Processor
from support.mixins import Sizing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Criterion"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Criteria(ntuple("Criteria", "variable threshold"), ABC):
    def __repr__(self): return f"{type(self).__name__}[{str(self.variable)}, {str(self.threshold)}]"
    def __call__(self, content, *args, stack=None, **kwargs):
        assert isinstance(stack, (list, type(None)))
        variable = self.variable if stack is None else tuple([self.variable] + stack)
        return self.execute(content, variable)

    @abstractmethod
    def execute(self, content, column): pass


class Floor(Criteria):
    def execute(self, content, variable): return content[variable] >= self.threshold

class Ceiling(Criteria):
    def execute(self, content, variable): return content[variable] <= self.threshold

class Null(Criteria):
    @typedispatcher
    def execute(self, content, variable): raise TypeError(type(content).__name__)
    @execute.register(pd.DataFrame)
    def dataframe(self, content, variable): return content[variable].notna()
    @execute.register(xr.Dataset)
    def dataset(self, content, variable): return content[variable].notnull()


class Criterion(object):
    FLOOR = Floor
    CEILING = Ceiling
    NULL = Null


class Filter(Processor, Sizing, title="Filtered"):
    def __init_subclass__(cls, *args, variables, query, **kwargs):
        assert isinstance(variables, list) and isinstance(query, str)
        cls.__variables__ = variables
        cls.__query__ = query

    def __init__(self, *args, criterion={},  **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(criterion, dict)
        assert all([issubclass(criteria, Criteria) for criteria in criterion.keys()])
        assert all([isinstance(parameter, (list, dict)) for parameter in criterion.values()])
        criterion = {criteria: parameters if isinstance(parameters, dict) else dict.fromkeys(parameters) for criteria, parameters in criterion.items()}
        criterion = [criteria(variable, threshold) for criteria, parameters in criterion.items() for variable, threshold in parameters.items()]
        self.__variables = self.__class__.__variables__
        self.__query = self.__class__.__query__
        self.__criterion = criterion

    def execute(self, contents, *args, **kwargs):
        query = str(contents[str(self.query)])
        variables = {variable: contents[variable] for variable in self.variables if variable in contents.keys()}
        variables = ODict(list(self.calculate(variables, *args, query=query, **kwargs)))
        if not bool(variables):
            return
        yield contents | dict(variables)

    def calculate(self, contents, *args, query, **kwargs):
        assert isinstance(contents, dict)
        for variable, content in contents.items():
            parameters = dict(variable=variable, query=query)
            prior = self.size(content)
            content = self.filter(content, *args, **parameters, **kwargs)
            post = self.size(content)
            self.notify(prior=prior, post=post, **parameters)
            if self.empty(content):
                return
            yield variable, content

    def filter(self, content, *args, **kwargs):
        mask = self.mask(content, *args, **kwargs)
        content = self.where(content, *args, mask=mask, **kwargs)
        return content

    def mask(self, content, *args, **kwargs):
        criterion = [criteria(content, *args, **kwargs) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        return mask

    def notify(self, *args, variable, query, prior, post, **kwargs):
        __logger__.info(f"Filter: {repr(self)}|{str(variable)}|{str(query)}[{prior:.0f}|{post:.0f}]")

    @typedispatcher
    def where(self, content, *args, mask=None, **kwargs): raise TypeError(type(content).__name__)
    @where.register(xr.Dataset)
    def where_dataset(self, dataset, *args, mask=None, **kwargs): return dataset.where(mask, drop=True) if bool(mask is not None) else dataset
    @where.register(pd.DataFrame)
    def where_dataframe(self, dataframe, *args, mask=None, **kwargs): return dataframe.where(mask).dropna(how="all", inplace=False) if bool(mask is not None) else dataframe

    @property
    def criterion(self): return self.__criterion
    @property
    def variables(self): return self.__variables
    @property
    def query(self): return self.__query



