# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import logging
import xarray as xr
from itertools import chain
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "equation", "source", "constant"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


def equation(variable, dataname, datatype, *args, domain, function, **kwargs):
    assert isinstance(domain, tuple) and callable(function)
    clsname = str(dataname).title()
    dataname = str(dataname).lower()
    attrs = dict(variable=variable, dataname=dataname, datatype=datatype, domain=domain, function=function)
    cls = type(clsname, (Equation,), {}, **attrs)
    yield cls

def source(variable, name, *args, position, parameters={}, variables={}, **kwargs):
    assert isinstance(variables, dict)
    title = lambda string: "|".join([str(substring).title() for substring in str(string).split("|")])
    varfunc = lambda string: ".".join([variable, string])
    locfunc = lambda string, fullname: "|".join([str(name), str(string)]).lower() if bool(fullname) else str(string).lower()
    for key, value in variables.items():
        Location = ntuple("Location", "source destination")
        location = Location(locfunc(value, kwargs.get("source", False)), locfunc(value, kwargs.get("destination", False)))
        clsname = title("|".join([name, value]))
        varname = varfunc(key)
        attrs = dict(variable=varname, position=position, location=location, parameters=parameters)
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
        return f"{repr(type(self))}[{nodes}]"

    def __str__(self):
        nodes = ", ".join([str(type(node)) for node in list(self.children)])
        return f"{str(type(self))}[{nodes}]"

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
    def __init_subclass__(cls, *args, position, location, parameters, **kwargs):
        assert isinstance(position, (int, str)) and isinstance(parameters, dict)
        cls.__position__ = position
        cls.__location__ = location
        cls.__parameters__ = parameters

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__position = self.__class__.__position__
        self.__location = self.__class__.__location__
        self.__parameters = self.__class__.__parameters__

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
        dataarray = dataarray.sel(self.parameters) if bool(self.parameters) else dataarray
        return dataarray

    def execute(self, order):
        wrapper = lambda *arrays: arrays[order.index(self)]
        wrapper.__name__ = str(self.name)
        return wrapper

    @property
    def position(self): return self.__position
    @property
    def location(self): return self.__location
    @property
    def parameters(self): return self.__parameters


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


class CalculationVariable(ABC):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if value is not None])
    def __hash__(self): return hash(tuple(self))

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])
    @classmethod
    def fields(cls): return list(cls._fields)

    def items(self): return list(zip(self.keys(), self.values()))
    def keys(self): return list(self._fields)
    def values(self): return list(self)


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
        function = lambda x: type(x) is CalculationMeta or issubclass(type(x), CalculationMeta)
        if not any([function(base) for base in bases]):
            return
        sources, equations, constants = {}, {}, {}
        for base in [base for base in reversed(bases) if type(base) is CalculationMeta]:
            sources.update(getattr(base, "__sources__", {}))
            equations.update(getattr(base, "__equations__", {}))
            constants.update(getattr(base, "__constants__", {}))
        sources = sources | cls.update("source", attrs)
        equations = equations | cls.update("equation", attrs)
        constants = constants | cls.update("constant", attrs)
        assert not set(sources.keys()) & set(constants.keys()) & set(equations.keys())
        cls.__sources__ = sources
        cls.__constants__ = constants
        cls.__equations__ = equations
        if not any([function(subbase) for base in bases for subbase in base.__bases__]):
            fields = kwargs.get("fields", [])
            variable_name = str(name).replace("Calculation", "Variable")
            variable_bases = (CalculationVariable, ntuple("Fields", fields))
            variable_attr = dict()
            cls.__variable__ = type(variable_name, variable_bases, variable_attr)
            cls.__registry__ = dict()
            cls.__fields__ = dict()
            return
        fields = [base.fields.items() for base in reversed(bases) if type(base) is CalculationMeta]
        fields = {field: value for (field, value) in chain(*fields)}
        update = {field: kwargs[field] for field in cls.variable.fields() if field in kwargs.keys()}
        cls.__fields__ = fields | update
        if len(cls.variable.fields()) == len(cls.fields):
            values = [cls.fields[field] for field in cls.variable.fields()]
            variable = cls.variable(*values)
            cls.registry[variable] = cls

    def __getitem__(cls, key): return cls.__registry__[key]
    def __iter__(cls): return iter(cls.__registry__.items())

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

    @staticmethod
    def update(name, attrs):
        update = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ == name]
        return {str(stage): stage for generator in iter(update) for stage in iter(generator)}

    @property
    def registry(cls): return cls.__registry__
    @property
    def variable(cls): return cls.__variable__
    @property
    def fields(cls): return cls.__fields__


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, *args, sources, constants, equations, **kwargs):
        self.__sources = sources
        self.__constants = constants
        self.__equations = equations

    def __getattr__(self, variable):
        if variable not in self.equations.keys():
            return super().__getattr__(variable)
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
        datasets = [dataset for dataset in generator if dataset is not None]
        return xr.merge(datasets) if bool(datasets) else None

    @abstractmethod
    def execute(self, dataset, *args, **kwargs): pass

    @property
    def sources(self): return self.__sources
    @property
    def constants(self): return self.__constants
    @property
    def equations(self): return self.__equations



