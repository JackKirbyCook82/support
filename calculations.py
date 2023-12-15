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
__all__ = ["Calculation", "equation", "source", "constant"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


def equation(variable, dataname, datatype, *args, domain, function, **kwargs):
    assert isinstance(domain, tuple) and callable(function)
    clsname = str(dataname).title()
    dataname = str(dataname).lower()
    attrs = dict(variable=variable, dataname=dataname, datatype=datatype, domain=domain, function=function)
    cls = type(clsname, (Equation,), {}, **attrs)
    yield cls

def source(variable, name, *args, position, variables={}, **kwargs):
    assert isinstance(variables, dict)
    title = lambda string: "|".join([str(substring).title() for substring in str(string).split("|")])
    varfunc = lambda string: ".".join([variable, string])
    locfunc = lambda optional, required, fullname: "|".join([str(name), str(value)]).lower() if bool(fullname) else str(value).lower()
    for key, value in variables.items():
        Location = ntuple("Location", "source destination")
        location = Location(locfunc(name, value, kwargs.get("source", False)), locfunc(name, value, kwargs.get("destination", False)))
        clsname = title("|".join([name, value]))
        varname = varfunc(key)
        attrs = dict(variable=varname, position=position, location=location)
        cls = type(clsname, (Source,), {}, **attrs)
        yield cls

def constant(variable, name, *args, position, **kwargs):
    clsname = "|".join([str(string).title() for string in str(name).split("|")])
    attrs = dict(variable=variable, position=position)
    cls = type(clsname, (Constant,), {}, **attrs)
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
    def __repr__(self):
        nodes = ', '.join([repr(type(node)) for node in list(self.children)])
        return "{}[{}]".format(repr(type(self)), nodes)

    def __str__(self):
        nodes = ", ".join([str(type(node)) for node in list(self.children)])
        return "{}[{}]".format(str(type(self)), nodes)

    def __init__(self, *args, variable, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variable = variable

    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)
    def __len__(self): return self.size

    @abstractmethod
    def execute(self, order): pass
    @property
    def variable(self): return self.__variable


class Equation(Stage):
    def __init_subclass__(cls, *args, dataname, datatype, domain, function, **kwargs):
        assert callable(function)
        cls.__dataname__ = dataname
        cls.__datatype__ = datatype
        cls.__function__ = function
        cls.__feeds__ = domain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dataname = self.__class__.__dataname__
        self.__datatype = self.__class__.__datatype__
        self.__function = self.__class__.__function__
        self.__feeds = self.__class__.__feeds__

    def __call__(self, *args, **kwargs):
        mapping = ODict([(stage, stage.locate(*args, **kwargs)) for stage in self.sources])
        order = list(mapping.keys())
        contents = list(mapping.values())
        execute = self.execute(order=order)
        dataarray = xr.apply_ufunc(execute, *contents, output_dtypes=[self.datatype], vectorize=True, dask="parallelized")
        dataset = dataarray.to_dataset(name=self.dataname)
        return dataset

    def execute(self, order):
        executes = [stage.execute(order) for stage in self.domain]
        wrapper = lambda *arrays: self.function(*[execute(*arrays) for execute in executes])
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def sources(self): return list(set(super().sources))
    @property
    def domain(self): return list(self.children)

    @property
    def dataname(self): return self.__dataname
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
        try:
            dataarray = self.locate(*args, **kwargs)
            dataset = dataarray.to_dataset(name=self.location.destination)
            return dataset
        except (KeyError, IndexError):
            return None

    def locate(self, *args, **kwargs):
        dataset = args[self.position] if isinstance(self.position, int) else kwargs[self.position]
        dataarray = dataset[self.location.source] if bool(self.location.source) else dataset
        return dataarray

    def execute(self, order):
        wrapper = lambda *arrays: arrays[order.index(self)]
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def position(self): return self.__position
    @property
    def location(self): return self.__location


class Constant(Stage):
    def __init_subclass__(cls, *args, position, **kwargs):
        assert isinstance(position, (int, str))
        cls.__position__ = position

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__position = self.__class__.__position__

    def locate(self, *args, **kwargs):
        value = args[self.position] if isinstance(self.position, int) else kwargs[self.position]
        return value

    def execute(self, order):
        wrapper = lambda *arrays: arrays[order.index(self)]
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def position(self): return self.__position


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        sources = [key for key, value in attrs.items() if isinstance(value, types.GeneratorType) and value.__name__ == "source"]
        constants = [key for key, value in attrs.items() if isinstance(value, types.GeneratorType) and value.__name__ == "constant"]
        equations = [key for key, value in attrs.items() if isinstance(value, types.GeneratorType) and value.__name__ == "equation"]
        exclude = sources + constants + equations
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        try:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        sources = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ == "source"]
        sources = {str(stage): stage for generator in iter(sources) for stage in iter(generator)}
        constants = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ == "constant"]
        constants = {str(stage): stage for generator in iter(constants) for stage in iter(generator)}
        equations = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ == "equation"]
        equations = {str(stage): stage for generator in iter(equations) for stage in iter(generator)}
        assert not set(sources.keys()) & set(constants.keys()) & set(equations.keys())
        cls.__sources__ = getattr(cls, "__sources__", {}) | sources
        cls.__constants__ = getattr(cls, "__constants__", {}) | constants
        cls.__equations__ = getattr(cls, "__equations__", {}) | equations

    def __call__(cls, *args, **kwargs):
        sources = {key: value(*args, **kwargs) for key, value in cls.__sources__.items()}
        constants = {key: value(*args, **kwargs) for key, value in cls.__constants__.items()}
        equations = {key: value(*args, **kwargs) for key, value in cls.__equations__.items()}
        stages = sources | constants | equations
        for stage in equations.values():
            for variable in stage.feeds:
                stage[variable] = stages[variable]
        stages = dict(sources=sources, constants=constants, equations=equations)
        instance = super(CalculationMeta, cls).__call__(*args, **stages, **kwargs)
        return instance


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, *args, sources, constants, equations, **kwargs):
        self.__sources = sources
        self.__constants = constants
        self.__equations = equations

    def __getattr__(self, variable):
        if variable not in self.equations.keys():
            raise AttributeError(variable)
        return self.equations[variable]

    def __getitem__(self, group):
        Locate = ntuple("Locate", "group variable stage")
        locate = [Locate(*str(key).split("."), value) for key, value in self.sources.items()]
        sources = ODict([(value.group, {}) for value in locate])
        for value in locate:
            sources[value.group][value.variable] = value.stage
        sources = sources[group]
        name = self.__class__.__name__
        Wrapper = ntuple(name, list(sources.keys()))
        wrapper = Wrapper(*list(sources.values()))
        return wrapper

    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        datasets = list(generator)
        return xr.merge(list(generator)) if bool(datasets) else None

    @abstractmethod
    def execute(self, dataset, *args, **kwargs): pass

    @property
    def sources(self): return self.__sources
    @property
    def constants(self): return self.__constants
    @property
    def equations(self): return self.__equations



