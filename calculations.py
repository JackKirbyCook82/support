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
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "equation", "source"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Position = ntuple("Position", "tag name")


def source(dataVar, dataName, *args, **kwargs):
    for dataKey, dataValue in kwargs.get("vars", {}).items():
        parameter = Position(dataKey, dataValue)
        variable = Position(dataVar, dataName)
        cls = type(".".join([dataVar, dataKey]), (Source,), {}, parameter=parameter, variable=variable)
        yield cls

def equation(dataVar, dataName, dataType, *args, domain, function, **kwargs):
    assert isinstance(domain, tuple) and callable(function)
    domain = [str(tags).split(".") if "." in tags else ["", tags] for tags in domain]
    domain = [(int(parameter) if str(parameter).isdigit() else str(parameter), str(variable)) for (parameter, variable) in domain]
    domain = [Locator(str(parameter) if bool(parameter) else None, str(variable)) for (parameter, variable) in domain]
    variable = Position(dataVar, dataName)
    cls = type(dataVar, (Equation,), {}, datatype=dataType, domain=domain, funciton=function, variable=variable)
    yield cls


class Locator(ntuple("Locator", "parameter variable")):
    pass


class StageMeta(ABCMeta):
    def __hash__(cls): return hash(cls.__locator__)
    def __init__(cls, *args, parameter, variable, **kwargs):
        cls.__locator__ = Locator(parameter, variable)


class SourceMeta(StageMeta):
    def __init__(cls, name, bases, attrs, *args, parameter=None, variable=None, **kwargs):
        if not any([type(base) is SourceMeta for base in bases]):
            pass
        super().__init__(*args, parameter=parameter, variable=variable, **kwargs)
        cls.__parameter__ = parameter
        cls.__variable__ = variable

    def __call__(cls, *args, **kwargs):
        attrs = dict(parameter=cls.__parameter__, variable=cls.__variable__)
        instance = super(SourceMeta, cls).__call__(*args, **attrs, **kwargs)
        return instance


class EquationMeta(StageMeta):
    def __init__(cls, name, bases, attrs, *args, datatype, function, variable, domain, **kwargs):
        if not any([type(base) is EquationMeta for base in bases]):
            pass
        super().__init__(*args, parameter=None, variable=variable, **kwargs)
        cls.__variable__ = variable
        cls.__function__ = function
        cls.__datatype__ = datatype
        cls.__domain__ = domain

    def __iter__(cls): return (locator for locator in cls.__domain__)
    def __call__(cls, *args, **kwargs):
        attrs = dict(variable=cls.__variable__, function=cls.__function__, datatype=cls.__datatype__)
        instance = super(EquationMeta, cls).__call__(*args, **attrs, **kwargs)
        return instance


class Stage(Node, ABC):
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


class Source(Stage, ABC, metaclass=SourceMeta):
    def __init__(self, *args, parameter, variable, **kwargs):
        super().__init__(*args, **kwargs)
        self.__parameter = parameter
        self.__variable = variable

    def locate(self, *args, **kwargs):
        pass

    def execute(self, order):
        wrapper = lambda *contents: contents[order.index(self)]
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def parameter(self): return self.__parameter
    @property
    def variable(self): return self.__variable


class Equation(Stage, ABC, metaclass=EquationMeta):
    def __init__(self, *args, variable, function, datatype, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variable = variable
        self.__function = function
        self.__datatype = datatype

    def __call__(self, *args, **kwargs):
        mapping = ODict([(stage, stage(*args, **kwargs)) for stage in self.sources])
        order = list(mapping.keys())
        contents = list(mapping.values())
        function = self.calculate(order=order)
        results = xr.apply_ufunc(function, *contents, output_dtypes=[self.datatype], vectorize=True, dask="parallelized")
        return results

    def execute(self, order):
        functions = [stage.execute(order) for stage in self.domain]
        wrapper = lambda *contents: self.function(*[function(*contents) for function in functions])
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def variable(self): return self.__variable
    @property
    def function(self): return self.__function
    @property
    def datatype(self): return self.__datatype


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
        variable = lambda key, value: type(key, (Source,), {}, parameter=Position(1, None), variable=Position(key, value))
        parameter = lambda key, value: type(key, (Source,), {}, parameter=Position(key, value), variable=None)
        sources = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ is "source"]
        sources = [value for generator in iter(sources) for value in iter(generator)]
        sources = sources + [variable(key, value) for key, value in kwargs.get("vars", {}).items()]
        sources = sources + [parameter(key, value) for key, value in kwargs.get("parms", {}).items()]
        equations = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ is "equation"]
        equations = [value for generator in iter(equations) for value in iter(generator)]
        assert not set(sources.keys()) & set(equations.keys())
        cls.__sources__ = getattr(cls, "__sources__", {}) | {hash(value): value for value in sources}
        cls.__equations__ = getattr(cls, "__equations__", {}) | {hash(value): value for value in equations}

    def __call__(cls, *args, **kwargs):
        sources = {key: value(*args, **kwargs) for key, value in cls.__sources__.items()}
        equations = {key: value(*args, **kwargs) for key, value in cls.__equations__.items()}
        stages = sources | equations
        for key, value in equations.items():
            for locator in iter(value):
                value[hash(locator)] = stages[hash(locator)]
        attrs = dict(sources=sources, equations=equations)
        instance = super(CalculationMeta, cls).__call__(*args, **attrs, **kwargs)
        return instance


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, *args, sources, equations, **kwargs):
        self.__sources = sources
        self.__equations = equations

    def __getattr__(self, attr):
        if attr in self.equations.keys():
            return self.equations[attr]
        elif attr in self.sources.keys():
            return self.sources[attr]
        return super().__getattr__(attr)

    @property
    def sources(self): return self.__sources
    @property
    def equations(self): return self.__equations



