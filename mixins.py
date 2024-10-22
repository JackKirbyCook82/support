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
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Node", "MultiNode", "Logging", "Emptying", "Sizing", "Function", "Generator", "Publisher", "Subscriber"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Node(object):
    def __init__(self, *args, **kwargs):
        self.__nodes = ODict()

    def set(self, key, value): self.nodes[key] = value
    def get(self, key): return self.nodes[key]

    def keys(self): return self.nodes.keys()
    def values(self): return self.nodes.values()
    def items(self): return self.nodes.items()

    def transverse(self):
        for value in self.values():
            yield value
            yield from value.transverse()

    @property
    def leafs(self): return [value for value in self.transverse() if not bool(value.children)]
    @property
    def branches(self): return [value for value in self.transverse() if bool(value.children)]
    @property
    def children(self): return list(self.nodes.values())
    @property
    def size(self): return len(self.nodes)

    @property
    def nodes(self): return self.__nodes


class MultiNode(Node):
    def get(self, key, index=None):
        if index is None: return self.nodes[key]
        return self.nodes[key][index]

    def set(self, key, value):
        if key not in self.nodes: self.nodes[key] = []
        if isinstance(value, list): self.nodes[key].extend(value)
        else: self.nodes[key].append(value)

    def transverse(self):
        for values in self.values():
            assert isinstance(values, list)
            for value in values:
                yield value
                yield from value.transverse()


class Emptying(object):
    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
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
    def empty_null(self, *args, **kwargs): return False


class Sizing(object):
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

    def __init__(self, *args, **kwargs):
        assert not inspect.isgeneratorfunction(self.execute)
        assert not inspect.isgeneratorfunction(self.function)

    def __call__(self, *args, source=None, **kwargs):
        return self.function(source, *args, **kwargs)

    @typedispatcher
    def function(self, source, *args, **kwargs): pass

    @function.register(types.NoneType)
    def function_empty(self, source, *args, **kwargs):
        assert isinstance(source, types.NoneType)
        return self.execute(*args, **kwargs)

    @function.register(types.FunctionType)
    def function_function(self, source, *args, **kwargs):
        content = source(*args, **kwargs)
        return self.execute(content, *args, **kwargs)

    @function.register(types.GeneratorType)
    def function_function(self, source, *args, **kwargs):
        return [self.execute(content, *args, **kwargs) for content in iter(source)]

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Generator(ABC):
    def __new__(cls, *args, **kwargs):
        execute = cls.execute
        if not inspect.isgeneratorfunction(execute):
            def wrapper(self, *arguments, **parameters):
                assert isinstance(self, cls)
                results = execute(self, *arguments, **parameters)
                if results is not None: yield results
            update_wrapper(wrapper, execute)
            setattr(cls, "execute", wrapper)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        assert inspect.isgeneratorfunction(self.generator)

    def __call__(self, *args, source=None, **kwargs):
        generator = self.generator(source, *args, **kwargs)
        yield from generator

    @typedispatcher
    def generator(self, source, *args, **kwargs):
        generator = self.execute(source, *args, **kwargs)
        yield from generator

    @generator.register(types.NoneType)
    def generator_empty(self, source, *args, **kwargs):
        assert isinstance(source, types.NoneType)
        generator = self.execute(*args, **kwargs)
        yield from generator

    @generator.register(types.FunctionType)
    def generator_function(self, source, *args, **kwargs):
        content = source(*args, **kwargs)
        generator = self.execute(content, *args, **kwargs)
        yield from generator

    @generator.register(types.GeneratorType)
    def generator_generator(self, source, *args, **kwargs):
        for content in iter(source):
            generator = self.execute(content, *args, **kwargs)
            yield from generator

    @abstractmethod
    def execute(self, *args, **kwargs): pass


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









