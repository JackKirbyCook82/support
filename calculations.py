# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import pandas as pd
import xarray as xr
from enum import Enum
from numbers import Number
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher, kwargsdispatcher
from support.meta import SingletonMeta
from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Equation", "Calculation"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Variable(Node, ABC):
    def __new__(cls, *args, function=None, locator=None, **kwargs):
        if cls is Variable and bool(function):
            return Dependent(*args, function=function, **kwargs)
        elif cls is Variable and bool(locator):
            return Source(*args, locator=locator, **kwargs)
        elif cls is Variable:
            return Constant(*args, **kwargs)
        return super().__new__(cls)

    def __str__(self): return self.key
    def __len__(self): return self.size
    def __init__(self, varkey, varname, vartype, *args, **kwargs):
        super().__init__(*args, name=varname, **kwargs)
        self.__type = vartype
        self.__key = varkey

    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)

    @property
    def constants(self): return [stage for stage in self.leafs if isinstance(stage, Constant)]
    @property
    def sources(self): return [stage for stage in self.leafs if isinstance(stage, Source)]
    @property
    def independents(self): return list(self.leafs)
    @property
    def dependents(self): return list(self.branches)
    @property
    def domain(self): return list(self.children)

    @property
    def arguments(self): return tuple([self.key, self.name, self.type])
    @property
    def parameters(self): return dict(formatter=self.formatter, style=self.style)
    def copy(self): return type(self)(*self.arguments, **self.parameters)

    @abstractmethod
    def calculate(self, execute, *args, datatype, sources, constants, **kwargs): pass
    @abstractmethod
    def execute(self, order): pass

    @property
    def type(self): return self.__type
    @property
    def key(self): return self.__key


class Independent(Variable):
    def __init__(self, *args, position, **kwargs):
        assert isinstance(position, (int, str, Enum))
        super().__init__(*args, **kwargs)
        self.__position = position

    def execute(self, order):
        calculation = lambda *contents: contents[order.index(self)]
        calculation.__name__ = "_".join([str(self.name).lower(), "independent", "calculation"])
        calculation.__order__ = [str(variable) for variable in order]
        return calculation

    def calculate(self, execute, *args, sources, constants, **kwargs):
        contents = list(sources) + list(constants)
        return execute(contents)

    @abstractmethod
    def locate(self, collection, *args, **kwargs): pass

    @property
    def parameters(self): return super().parameters | dict(position=self.position)
    @property
    def position(self): return self.__position


class Source(Independent):
    def __init__(self, *args, locator, **kwargs):
        assert isinstance(locator, str)
        super().__init__(*args, **kwargs)
        self.__locator = locator

    @typedispatcher
    def locate(self, collection): raise TypeError(type(collection).__name__)
    @locate.register(dict)
    def mapping(self, mapping): return mapping[self.position][self.locator]
    @locate.register(xr.Dataset, pd.DataFrame)
    def collection(self, collection): return collection[self.locator]

    @property
    def location(self):
        def wrapper(collection):
            results = self.locate(collection)
            assert isinstance(results, (xr.DataArray, pd.Series))
            results = results.rename(self.name)
            return results
        wrapper.__name__ = "_".join([str(self.name), "location"])
        return wrapper

    @property
    def parameters(self): return super().parameters | dict(locator=self.locator)
    @property
    def locator(self): return self.__locator


class Constant(Independent):
    def locate(self, *args, **kwargs): return kwargs[self.position]


class Dependent(Variable):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(function, types.LambdaType)
        self.__function = function

    def execute(self, order):
        domains = [variable.execute(order) for variable in self.domain]
        calculation = lambda *contents: self.function(*[domain(*contents) for domain in domains])
        calculation.__name__ = "_".join([str(self.name).lower(), "dependent", "calculation"])
        calculation.__order__ = [str(variable) for variable in order]
        return calculation

    @kwargsdispatcher("datatype")
    def calculate(self, execute, *args, **kwargs): raise TypeError(kwargs["datatype"].__name__)

    @calculate.register.value(xr.DataArray)
    def dataarray(self, execute, *args, sources, constants, **kwargs):
        assert all([isinstance(source, xr.DataArray) for source in sources])
        contents = list(sources) + list(constants)
        dataarray = xr.apply_ufunc(execute, *contents, output_dtypes=[self.type], vectorize=True)
        return dataarray

    @calculate.register.value(pd.Series)
    def series(self, execute, *args, sources, constants, **kwargs):
        assert all([isinstance(source, pd.Series) for source in sources])
        function = lambda array, *arguments: execute(*list(array), *list(arguments))
        series = pd.concat(list(sources), axis=1).apply(function, axis=1, raw=False, args=tuple(constants))
        return series

    @property
    def calculation(self):
        def wrapper(collections, *args, **kwargs):
            sources = ODict([(stage, stage.locate(collections)) for stage in self.sources])
            constants = ODict([(stage, stage.locate(*args, **kwargs)) for stage in self.constants])
            order = list(sources.keys()) + list(constants.keys())
            execute = self.execute(order)
            sources = list(sources.values())
            constants = list(constants.values())
            assert isinstance(sources, list) and isinstance(constants, list)
            assert all([isinstance(constant, Number) for constant in constants])
            datatype = set([type(content) for content in sources])
            assert len(datatype) == 1
            datatype = list(datatype)[0]
            parameters = dict(datatype=datatype, sources=sources, constants=constants)
            results = self.calculate(execute, *args, **parameters, **kwargs)
            assert isinstance(results, (xr.DataArray, pd.Series))
            results = results.rename(self.name)
            return results
        wrapper.__name__ = "_".join([str(self.name), "location"])
        return wrapper

    @property
    def parameters(self): return super().parameters | dict(function=self.function)
    @property
    def function(self): return self.__function


class EquationMeta(SingletonMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.items() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __iter__(cls): return iter(cls.__variables__.items())
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = {key: variable.copy() for key, variable in getattr(cls, "__variables__", {}).items()}
        updated = {str(variable): variable for variable in attrs.values() if isinstance(variable, Variable)}
        cls.__variables__ = existing | updated

    def __call__(cls, *args, **kwargs):
        variables = ODict(list(iter(cls)))
        for variable in variables.values():
            if isinstance(variable, Dependent):
                for key in list(inspect.signature(variable.function).parameters.keys()):
                    variable[key] = variables[key]
        instance = super(EquationMeta, cls).__call__(*args, variables=variables, **kwargs)
        return instance


class Equation(ABC, metaclass=EquationMeta):
    def __iter__(self): return list(self.variables.items())
    def __init__(self, *args, variables, **kwargs):
        assert isinstance(variables, dict)
        assert all([isinstance(variable, Variable) for variable in variables.values()])
        self.__variables = variables

    def __getattr__(self, variable):
        if variable not in self.variables.keys():
            raise AttributeError(variable)
        variable = self.variables[variable]
        return variable.location if isinstance(variable, Independent) else variable.calculation

    @property
    def variables(self): return self.__variables


class Fields(frozenset):
    def __getitem__(self, key): return self.todict()[key]
    def __bool__(self): return None not in self.values()

    def __new__(cls, contents):
        assert isinstance(contents, (dict, list))
        contents = ODict.fromkeys(contents) if isinstance(contents, list) else contents
        contents = list(contents.items())
        return super().__new__(cls, contents)

    def __or__(self, contents):
        assert isinstance(contents, (dict, list))
        contents = ODict.fromkeys(contents) if isinstance(contents, list) else contents
        contents = self.todict() | contents
        return type(self)(contents)

    def todict(self): return ODict(list(self))
    def tolist(self): return list(self)
    def keys(self): return self.todict().keys()
    def values(self): return self.todict().values()
    def items(self): return self.todict().items()


class CalculationMeta(ABCMeta):
    def __iter__(cls): return iter(list(cls.registry.items()))
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is CalculationMeta for base in cls.__bases__]):
            return
        if not any([type(subbase) is CalculationMeta for base in cls.__bases__ for subbase in base.__bases__]):
            cls.__fields__ = Fields(list(set(kwargs.get("fields", []))))
            cls.__registry__ = ODict()
            return
        fields = cls.fields | {key: kwargs[key] for key in cls.fields.keys() if key in kwargs.keys()}
        if bool(fields):
            cls.registry[fields] = cls
        cls.__equation__ = kwargs.get("equation", getattr(cls, "__equation__", None))
        cls.__fields__ = fields

    @property
    def registry(cls): return cls.__registry__
    @property
    def fields(cls): return cls.__fields__


class Calculation(ABC, metaclass=CalculationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, **kwargs):
        self.__equation = self.__class__.__equation__

    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        contents = list(generator)
        content = self.combine(contents[0], *contents[1:])
        return content

    @typedispatcher
    def combine(self, content, *contents): raise TypeError(type(content).__name__)
    @combine.register(xr.DataArray)
    def dataarray(self, content, *contents): return xr.merge([content] + list(contents))
    @combine.register(pd.Series)
    def series(self, content, *contents): return pd.concat([content] + list(contents), axis=1)

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def equation(self): return self.__equation



