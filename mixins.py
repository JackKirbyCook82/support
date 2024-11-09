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
__all__ = ["Logging", "Emptying", "Sizing", "Carryover", "Function", "Generator", "Publisher", "Subscriber"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


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


class Carryover(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: pass
        carryover = kwargs.get("carryover", getattr(cls, "carryover", []))
        leading = kwargs.get("leading", getattr(cls, "leading", True))
        assert isinstance(carryover, (list, str)) and isinstance(leading, bool)
        cls.carryover = [carryover] if isinstance(carryover, str) else carryover
        cls.leading = leading

    def __new__(cls, *args, **kwargs):
        execute = cls.execute
        signature = list(inspect.signature(execute).parameters.keys())
        indexes = [cls.arguments(signature).index(argument) for argument in cls.carryover]

        if not inspect.isgeneratorfunction(execute):
            def wrapper(self, *arguments, **parameters):
                assert isinstance(self, cls)
                carryover = [arguments[index] for index in indexes]
                result = execute(self, *arguments, **parameters)
                return *carryover, result if bool(cls.leading) else result, *carryover

        elif inspect.isgeneratorfunction(execute):
            def wrapper(self, *arguments, **parameters):
                assert isinstance(self, cls)
                carryover = [arguments[index] for index in indexes]
                for result in execute(self, *arguments, **parameters):
                    yield *carryover, result if bool(cls.leading) else result, *carryover

        update_wrapper(wrapper, execute)
        setattr(cls, "execute", wrapper)
        try: return super().__new__(cls, *args, **kwargs)
        except TypeError: return super().__new__(cls)

    @staticmethod
    def parameters(signature): return signature[signature.index("args")+1:signature.index("kwargs")]
    @staticmethod
    def arguments(signature): return signature[signature.index("self")+1:signature.index("args")]

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Function(ABC):
    def __new__(cls, *args, **kwargs):
        if not inspect.isgeneratorfunction(cls.execute):
            return super().__new__(cls)
        execute = cls.execute

        def wrapper(self, *arguments, **parameters):
            assert isinstance(self, cls)
            generator = execute(self, *arguments, **parameters)
            return list(generator)

        update_wrapper(wrapper, execute)
        setattr(cls, "execute", wrapper)
        mro = list(cls.__mro__)
        assert Generator not in mro
        if Carryover in mro: assert mro.index(Carryover) > mro.index(Function)
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
        if Carryover in mro: assert mro.index(Carryover) > mro.index(Generator)
        try: return super().__new__(cls, *args, **kwargs)
        except TypeError: return super().__new__(cls)

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Logging(object):
    def __repr__(self): return str(self.name)
    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: pass
        self.__name = kwargs.pop("name", self.__class__.__name__)
        self.__logger = __logger__

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


class Publisher(object):
    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: pass
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
        except TypeError: pass
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









