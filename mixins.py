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
__all__ = ["AttributeNode", "Logging", "Emptying", "Sizing", "Function", "Generator", "Publisher", "Subscriber"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class AttributeNode(object):
    def __new__(cls, mapping):
        assert isinstance(mapping, dict)
        instance = super().__new__(cls)
        for attribute, content in mapping.items():
            assert isinstance(attribute, str)
            setattr(instance, attribute, content)
        return instance


class Emptying(object):
    @typedispatcher
    def empty(self, content): raise TypeError(type(content))
    @empty.register(dict)
    def empty_mapping(self, mapping): return all([self.empty(value) for value in mapping.values()]) if bool(mapping) else True
    @empty.register(list)
    def empty_collection(self, collection): return all([self.empty(value) for value in collection]) if bool(collection) else True
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)
    @empty.register(types.NoneType)
    def empty_nothing(self, *args, **kwargs): return True


class Sizing(object):
    @typedispatcher
    def size(self, content): raise TypeError(type(content))
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
    @size.register(types.NoneType)
    def size_nothing(self, *args, **kwargs): return 0


class Function(ABC):
    def __new__(cls, *args, **kwargs):
        execute = cls.execute
        if inspect.isgeneratorfunction(execute):
            def wrapper(self, *arguments, **parameters):
                assert isinstance(self, cls)
                generator = execute(self, *arguments, **parameters)
                return list(generator)
            update_wrapper(wrapper, execute)
            setattr(cls, "execute", wrapper)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs): assert not inspect.isgeneratorfunction(self.execute)
    def __call__(self, *args, **kwargs): return self.execute(*args, **kwargs)

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Generator(ABC):
    def __new__(cls, *args, **kwargs):
        generator = cls.generator
        if not inspect.isgeneratorfunction(generator):
            def wrapper(self, *arguments, **parameters):
                assert isinstance(self, cls)
                results = generator(self, *arguments, **parameters)
                if results is not None: yield results
            update_wrapper(wrapper, generator)
            setattr(cls, "generator", wrapper)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs): assert inspect.isgeneratorfunction(self.generator)
    def __call__(self, *args, **kwargs): yield from self.generator(*args, **kwargs)

    @abstractmethod
    def generator(self, *args, **kwargs): pass


class Logging(object):
    def __repr__(self): return str(self.name)
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.pop("name", self.__class__.__name__)
        self.__logger = __logger__

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


class Publisher(object):
    def __init__(self, *args, **kwargs):
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









