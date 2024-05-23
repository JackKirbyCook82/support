# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   Query Object
@author: Jack Kirby Cook

"""

import inspect
import pandas as pd
import xarray as xr
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Header", "Query"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class Header(ntuple("Header", "datatype index columns")):
    def __call__(self, content): return self.execute(content)
    def __new__(cls, datatype, *args, index, columns, **kwargs):
        assert isinstance(index, list) and isinstance(columns, list)
        return super().__new__(cls, datatype, index, columns)

    @typedispatcher
    def execute(self, content): raise ValueError(type(content).__name__)

    @execute.register(pd.DataFrame)
    def dataframe(self, dataframe):
        pass

    @execute.register(xr.Dataset)
    def dataset(self, dataset):
        pass


class QueryMeta(type):
    def __call__(cls, *args, **kwargs):
        def decorator(execute):
            assert callable(execute)
            assert inspect.isgeneratorfunction(execute)
            instance = super(QueryMeta, cls).__call__(execute, *args, **kwargs)
            return instance
        return decorator


class Query(object, metaclass=QueryMeta):
    def __init__(self, execute, arguments=[], parameters={}, headers={}):
        assert isinstance(arguments, list) and isinstance(parameters, dict)
        assert all([isinstance(parameter, list) for parameter in parameters.values])
        self.__parameters = parameters
        self.__arguments = arguments
        self.__headers = headers
        self.__execute = execute

    def __call__(self, query, *args, **kwargs):
        isolated = self.isolate(query)
        calculated = self.execute(*args, **isolated, **kwargs)
        parsed = self.parser(calculated)
        return query | isolated | calculated | parsed

    def isolate(self, query):
        assert isinstance(query, dict)
        arguments = {argument: query.get(argument, None) for argument in self.arguments}
        parameters = {parameter: {content: query.get(content, None) for content in contents} for parameter, contents in self.parameters.items()}
        return arguments | parameters

    def parse(self, calculated):
        assert isinstance(calculated, dict)
        calculated = {key: value for key, value in calculated.items() if key in self.header.keys()}
        parsed = {key: self.headers[key](value) for key, value in calculated.items()}
        return parsed

    @property
    def parameters(self): return self.__parameters
    @property
    def arguments(self): return self.__arguments
    @property
    def execute(self): return self.__execute







