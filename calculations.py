# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import inspect
import logging
import xarray as xr
from numbers import Number
from abc import ABC, ABCMeta
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.mixins import Node
from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "feed", "equation"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


def feed(dataname, datatype, *args, **kwargs):
    name = str(dataname) + "Feed"
    axes = str(kwargs.get("axes", "")).strip("()").split(",")
    variable = Variable(dataname, datatype, axes)
    values = [kwargs.get(field, None) for field in Locator._fields]
    locator = Locator(*values)
    cls = type(name, (Feed,), {}, variable=variable, locator=locator)
    return cls

def equation(dataname, datatype, *args, function, **kwargs):
    name = str(dataname) + "Equation"
    axes = str(kwargs.get("axes", "")).strip("()").split(",")
    variable = Variable(dataname, datatype, axes)
    cls = type(name, (Equation,), {}, variable=variable, function=function)
    return cls


class Variable(ntuple("Variable", "name type axes")): pass
class Locator(ntuple("Locator", "key index variable")): pass


class Stage(Node, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))

    def __init__(self, *args, index, **kwargs):
        super().__init__(*args, **kwargs)
        self.__index = index

    def __repr__(self): return str(self.tree)
    def __len__(self): return self.size

    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)

    @property
    def sources(self): return sorted(super().sources, key=lambda x: x.index)
    @property
    def domain(self): return self.children

    @property
    def name(self): return self.variable.name
    @property
    def type(self): return self.variable.type
    @property
    def axes(self): return self.variable.axes

    @property
    def variable(self): return self.__class__.__variable__
    @property
    def index(self): return self.__index


class Feed(Stage):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__locator__ = kwargs.get("locator", getattr(cls, "__locator__", None))
        super().__init_subclass__(*args, **kwargs)

    def __call__(self, feeds):
        assert isinstance(feeds, (list, dict, xr.DataArray, xr.Dataset, Number))
        return self.locate(feeds)

    def calculate(self, order):
        wrapper = lambda *contents: contents[order.index(self)]
        wrapper.__name__ = str(self.name)
        return wrapper

    @typedispatcher
    def locate(self, feeds): raise TypeError(type(feeds).__name__)
    @locate.register(dict)
    def mapping(self, mapping): return self.locate(mapping[self.locator.key])
    @locate.register(list)
    def contents(self, contents): return self.locate(contents[self.locator.index])
    @locate.register(xr.Dataset)
    def dataset(self, dataset): return self.locate(dataset[self.locator.variable])
    @locate.register(xr.DataArray)
    def dataarray(self, dataarray): return dataarray
    @locate.register(Number)
    def value(self, value): return value

    @property
    def locator(self): return self.__class__.__locator__


class Equation(Stage):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__function__ = kwargs.get("function", getattr(cls, "__function__", None))
        super().__init_subclass__(*args, **kwargs)

    def __call__(self, feeds):
        contents = ODict([(source, source.locate(feeds)) for source in self.sources])
        function = self.calculate(list(contents.keys()))
        results = xr.apply_ufunc(function, *list(contents.values()), output_dtypes=[self.type], vectorize=True, dask="parallelized")
        return results

    def calculate(self, order):
        calculations = [stage.calculate(order) for stage in list(self.domain)]
        wrapper = lambda *contents: self.function(*[calculation(*contents) for calculation in calculations])
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def function(self): return self.__class__.__function__


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        feeds = [key for key, value in attrs.items() if inspect.isclass(value) and issubclass(value, Feed)]
        equations = [key for key, value in attrs.items() if inspect.isclass(value) and issubclass(value, Equation)]
        exclude = feeds + equations
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        try:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        cls.__feeds__ = {key: value for key, value in getattr(cls, "__feeds__", {}).items()}
        cls.__equations__ = {key: value for key, value in getattr(cls, "__equations__", {}).items()}
        feeds = {key: value for key, value in attrs.items() if inspect.isclass(value) and issubclass(value, Feed)}
        equations = {key: value for key, value in attrs.items() if inspect.isclass(value) and issubclass(value, Equation)}
        cls.__feeds__.update(feeds)
        cls.__equations__.update(equations)

    def __call__(cls, *args, **kwargs):
        equations = {key: value(*args, index=index, **kwargs) for index, (key, value) in enumerate(cls.__equations__.items())}
        feeds = {key: value(*args, index=index, **kwargs) for index, (key, value) in enumerate(cls.__feeds__.items())}
        for value in equations.values():
            for key in list(inspect.getfullargspec(value.function).args):
                value[key] = equations[key] if key in equations.keys() else feeds[key]
        instance = super(CalculationMeta, cls).__call__(equations, feeds, *args, **kwargs)
        return instance


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, equations, feeds, *args, **kwargs):
        self.__equations = equations
        self.__feeds = feeds

    def __getattr__(self, attr):
        if attr in self.equations.keys():
            return self.equations[attr]
        elif attr in self.feeds.keys():
            return self.feeds[attr]
        raise AttributeError(attr)

    @property
    def equations(self): return self.__equations
    @property
    def feeds(self): return self.__feeds



