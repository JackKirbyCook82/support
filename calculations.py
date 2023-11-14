# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import logging
import xarray as xr
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "equation", "source"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


def equation(variable, name, datatype, *args, domain, function, **kwargs):
    assert isinstance(domain, tuple) and callable(function)
    title = str(name).title()
    attrs = dict(variable=variable, datatype=datatype, feeds=domain, function=function)
    cls = type(title, (Equation,), {}, **attrs)
    yield cls

def source(variable, name, *args, position, variables={}, **kwargs):
    assert isinstance(variables, dict)
    for key, value in variables.items():
        title = [str(string).title() for string in "|".join(name, key).split("|")]
        create = lambda subvariable: ".".join([variable, subvariable])
        attrs = dict(variable=create(variable, key), position=position, location=value)
        cls = type(title, (Source,), {}, **attrs)
        yield cls
    if not variables:
        title = [str(string).title() for string in str(name).split("|")]
        attrs = dict(variable=variable, position=position, location=None)
        cls = type(title, (Source,), {}, **attrs)
        yield cls


class StageMeta(ABCMeta):
    def __repr__(cls): return cls.__name__
    def __str__(cls): return cls.__variable__

    def __init__(cls, *args, **kwargs):
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__variable__ is not None
        formatter = lambda key, node: str(node.variable)
        name = str(cls.__name__).lower()
        parameters = dict(formatter=formatter, name=name, variable=cls.__variable__)
        instance = super(StageMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Stage(Node, metaclass=StageMeta):
    def __init__(self, *args, variable, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variable = variable

    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)
    def __repr__(self): return str(self.tree)
    def __len__(self): return self.size

    @abstractmethod
    def execute(self, order): pass

    @property
    def sources(self): return list(set(super().sources))
    @property
    def domain(self): return list(self.children)
    @property
    def variable(self): return self.__variable


class Equation(Stage):
    def __init_subclass__(cls, *args, datatype, feeds, function, **kwargs):
        assert callable(function)
        cls.__datatype__ = datatype
        cls.__function__ = function
        cls.__feeds__ = feeds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__datatype = self.__class__.__datatype__
        self.__function = self.__class__.__function__
        self.__feeds = self.__class__.__feeds__

    def __call__(self, *args, **kwargs):
        mapping = ODict([(stage, stage(*args, **kwargs)) for stage in self.sources])
        order = list(mapping.keys())
        contents = list(mapping.values())
        execute = self.execute(order=order)
        dataarray = xr.apply_ufunc(execute, *contents, output_dtypes=[self.datatype], vectorize=True, dask="parallelized")
#        dataset = dataarray.to_dataset(name=)
        return dataarray

    def execute(self, order):
        executes = [stage.execute(order) for stage in self.domain]
        wrapper = lambda *contents: self.function(*[execute(*contents) for execute in executes])
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def datatype(self): return self.__datatype
    @property
    def function(self): return self.__function
    @property
    def feeds(self): return self.__feeds


class Source(Stage):
    def __init_subclass__(cls, *args, position, location, **kwargs):
        assert isinstance(position, (int, str))
        cls.__position__ = position
        cls.__location__ = location

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__position = self.__class__.__position__
        self.__location = self.__class__.__location__

    def __call__(self, *args, **kwargs):
        content = args[self.position] if isinstance(self.position, int) else kwargs[self.position]
        content = content[self.location] if bool(self.location) else content
        return content

    def execute(self, order):
        wrapper = lambda *contents: contents[order.index(self)]
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def position(self): return self.__position
    @property
    def location(self): return self.__location


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        sources = [key for key, value in attrs.items() if isinstance(value, types.GeneratorType) and value.__name__ is "source"]
        equations = [key for key, value in attrs.items() if isinstance(value, types.GeneratorType) and value.__name__ is "equation"]
        exclude = sources + equations
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        try:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        sources = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ is "source"]
        sources = {str(stage): stage for generator in iter(sources) for stage in iter(generator)}
        equations = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ is "equation"]
        equations = {str(stage): stage for generator in iter(equations) for stage in iter(generator)}
        assert not set(sources.keys()) & set(equations.keys())
        cls.__sources__ = getattr(cls, "__sources__", {}) | sources
        cls.__equations__ = getattr(cls, "__equations__", {}) | equations

    def __call__(cls, *args, **kwargs):
        sources = {key: value(*args, **kwargs) for key, value in cls.__sources__.items()}
        equations = {key: value(*args, **kwargs) for key, value in cls.__equations__.items()}
        stages = sources | equations
        for instance in equations.values():
            for variable in instance.feeds:
                instance[variable] = stages[variable]
        stages = dict(sources=sources, equations=equations)
        instance = super(CalculationMeta, cls).__call__(*args, **stages, **kwargs)
        return instance


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, *args, sources, equations, **kwargs):
        self.__sources = sources
        self.__equations = equations

    def __getattr__(self, variable):
        if variable not in self.equations.keys():
            raise AttributeError(variable)
        return self.equations[variable]

    def __getitem__(self, variable):
        if variable not in self.sources.keys():
            raise KeyError(variable)

#        wrapper = ntuple()
#        return wrapper

    def __call__(self, *args, **kwargs):
        dataset = xr.Dataset()
        self.execute(dataset, *args, **kwargs)
        return dataset

    @abstractmethod
    def execute(self, dataset, *args, **kwargs): pass

    @property
    def sources(self): return self.__sources
    @property
    def equations(self): return self.__equations



