# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   Query Object
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
import xarray as xr

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Data", "Header", "Query"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class QueryMeta(type):
    def __call__(cls, *args, **kwargs):
        def decorator(generator):
            assert isinstance(generator, types.GeneratorType)
            instance = super(QueryMeta, cls).__call__(generator, *args, **kwargs)
            return instance
        return decorator


class Query(object, metaclass=QueryMeta):
    def __init__(self, generator, *inlet, **outlet):
        assert all([isinstance(argument, str) for argument in inlet])
        assert all([isinstance(parameter, Header) for parameter in outlet.values()])
        self.__generator = generator
        self.__outlet = outlet
        self.__inlet = inlet

    def __call__(self, contents, *args, **kwargs):
        assert isinstance(contents, dict)
        feed = filter(None, [contents.get(argument, None) for argument in self.inlet])
        generator = self.generator(*feed, *args, **kwargs)
        for updated in iter(generator):
            assert isinstance(updated, dict)
            parsed = {parameter: header(updated.pop(parameter, None)) for parameter, header in self.outlet.items()}
            parsed = {parameter: content for parameter, content in parsed.items() if content is not None}
            yield contents | updated | parsed

    @property
    def generator(self): return self.__generator
    @property
    def outlet(self): return self.__outlet
    @property
    def inlet(self): return self.__inlet


class Header(object):
    pass


class Data(object):
    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(dict)
    def empty_mapping(self, mapping): return all([self.empty(value) for value in mapping.values()]) if bool(mapping) else False
    @empty.register(list)
    def empty_collection(self, collection): return all([self.empty(value) for value in collection]) if bool(collection) else False
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)

    @typedispatcher
    def size(self, content): raise TypeError(type(content).__name__)
    @size.register(dict)
    def size_mapping(self, mapping): return sum([self.size(value) for value in mapping.values()])
    @size.register(list)
    def size_collection(self, collection): return sum([self.size(value) for value in collection])
    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.DataFrame)
    def size_dataframe(self, dataframe): return len(dataframe.dropna(how="all", inplace=False).index)
    @size.register(pd.Series)
    def size_series(self, series): return len(series.dropna(how="all", inplace=False).index)



