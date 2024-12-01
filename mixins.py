# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Mixins Object
@author: Jack Kirby Cook

"""

import types
import inspect
import logging
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from functools import update_wrapper

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Logging", "Emptying", "Memory", "Sizing", "Function", "Generator", "Pivoting", "Sourcing", "Publisher", "Subscriber"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Emptying(object):
    @typedispatcher
    def empty(self, content, *args, **kwargs): raise TypeError(type(content))
    @empty.register(dict)
    def __mapping(self, mapping, *args, **kwargs): return all([self.empty(value, *args, **kwargs) for value in mapping.values()]) if bool(mapping) else True
    @empty.register(list)
    def __collection(self, collection, *args, **kwargs): return all([self.empty(value, *args, **kwargs) for value in collection]) if bool(collection) else True
    @empty.register(xr.Dataset)
    def __dataset(self, dataset, key, *args, **kwargs): return self.empty([dataset[key]], *args, **kwargs)
    @empty.register(xr.DataArray)
    def __dataarray(self, dataarray, *args, **kwargs): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, **kwargs): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def __series(self, series, *args, **kwargs): return bool(series.empty)
    @empty.register(types.NoneType)
    def __nothing(self, *args, **kwargs): return True


class Sizing(object):
    @typedispatcher
    def size(self, content, *args, **kwargs): raise TypeError(type(content))
    @size.register(dict)
    def __mapping(self, mapping, *args, **kwargs): return sum([self.size(value) for value in mapping.values()])
    @size.register(list)
    def __collection(self, collection, *args, **kwargs): return sum([self.size(value) for value in collection])
    @size.register(xr.Dataset)
    def __dataset(self, dataset, key, *args, **kwargs): return self.size([dataset[key]], *args, **kwargs)
    @size.register(xr.DataArray)
    def __dataarray(self, dataarray, *args, **kwargs): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, **kwargs): return len(dataframe.dropna(how="all", inplace=False).index)
    @size.register(pd.Series)
    def __series(self, series, *args, **kwargs): return len(series.dropna(how="all", inplace=False).index)
    @size.register(types.NoneType)
    def __nothing(self, *args, **kwargs): return 0


class Memory(object):
    @typedispatcher
    def memory(self, content, *args, **kwargs): raise TypeError(type(content))
    @memory.register(dict)
    def __mapping(self, mapping, *args, **kwargsg): return sum([self.memory(value) for value in mapping.values()])
    @memory.register(list)
    def __collection(self, collection, *args, **kwargs): return sum([self.memory(value) for value in collection])
    @memory.register(pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    def __content(self, content, *args, **kwargs): return content.nbytes
    @memory.register(types.NoneType)
    def __nothing(self, *args, **kwargs): return 0


class Function(ABC):
    def __new__(cls, *args, combine=None, **kwargs):
        if not inspect.isgeneratorfunction(cls.execute):
            return super().__new__(cls)
        assert callable(combine) or isinstance(combine, types.NoneType)
        execute = cls.execute

        def wrapper(self, *arguments, **parameters):
            assert isinstance(self, cls)
            generator = execute(self, *arguments, **parameters)
            collection = list(generator)
            if not bool(collection): return
            if combine is None: return collection
            else: return combine(collection)

        update_wrapper(wrapper, execute)
        setattr(cls, "execute", wrapper)
        mro = list(cls.__mro__)
        assert Generator not in mro
        try: return super().__new__(cls, *args, **kwargs)
        except TypeError: return super().__new__(cls)

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Generator(ABC):
    def __new__(cls, *args, **kwargs):
        if inspect.isgeneratorfunction(cls.execute):
            return super().__new__(cls)
        execute = cls.execute

        def wrapper(self, *arguments, **parameters):
            assert isinstance(self, cls)
            results = execute(self, *arguments, **parameters)
            if results is not None: yield results

        update_wrapper(wrapper, execute)
        setattr(cls, "execute", wrapper)
        mro = list(cls.__mro__)
        assert Function not in mro
        try: return super().__new__(cls, *args, **kwargs)
        except TypeError: return super().__new__(cls)

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Sourcing(object):
    @typedispatcher
    def source(self, content, *args, query, **kwargs): raise TypeError(type(content))

    @typedispatcher.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, query, **kwargs):
        generator = dataframe.groupby(list(query))
        for values, dataframe in iter(generator):
            yield query(list(values)), dataframe

    @typedispatcher.register(xr.Dataset)
    def __dataset(self, dataset, *args, query, **kwargs):
        for field in list(query):
            dataset = dataset.expand_dims(field)
        dataset = dataset.stack(stack=list(query))
        generator = dataset.groupby("stack")
        for values, dataset in iter(generator):
            dataset = dataset.unstack().drop_vars("stack")
            yield query(list(values)), dataset


class Pivoting(object):
    @staticmethod
    def pivot(dataframe, *args, stacking=[], by, **kwargs):
        assert isinstance(dataframe, pd.DataFrame) and isinstance(stacking, list) and isinstance(by, str)
        index = set(dataframe.columns) - ({by} | set(stacking))
        dataframe = dataframe.pivot(index=list(index), columns=["scenario"])
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @staticmethod
    def unpivot(dataframe, *args, unstacking=[], by, **kwargs):
        assert isinstance(dataframe, pd.DataFrame) and isinstance(unstacking, list) and isinstance(by, str)
        index = set(dataframe.columns) - ({by} | set(unstacking))
        dataframe = dataframe.set_index(list(index), drop=True, inplace=False)
        dataframe = dataframe.stack()
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class Logging(object):
    def __repr__(self): return str(self.name)
    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.__name = kwargs.pop("name", self.__class__.__name__)
        self.__logger = __logger__

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


class Publisher(object):
    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__subscribers = set()

    def register(self, *subscribers):
        assert all([isinstance(subscriber, Subscriber) for subscriber in subscribers])
        for subscriber in subscribers:
            self.subscribers.add(subscriber)
            subscriber.publishers.add(self)

    def unregister(self, *subscribers):
        assert all([isinstance(subscriber, Subscriber) for subscriber in subscribers])
        for subscriber in subscribers:
            self.subscribers.discard(subscriber)
            subscriber.publishers.discard(self)

    def publish(self, event, *args, **kwargs):
        for subscriber in self.subscribers:
            subscriber.observe(event, *args, publisher=self, **kwargs)

    @property
    def subscribers(self): return self.__subscribers
    @property
    def name(self): return self.__name


class Subscriber(ABC):
    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__publishers = set()

    def observe(self, event, *args, publisher, **kwargs):
        assert isinstance(publisher, Publisher)
        self.reaction(event, *args, publisher=publisher, **kwargs)

    @abstractmethod
    def reaction(self, event, *args, publisher, **kwargs): pass
    @property
    def publishers(self): return self.__publishers
    @property
    def name(self): return self.__name









