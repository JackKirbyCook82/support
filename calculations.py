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
from itertools import product
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher, kwargsdispatcher
from support.pipelines import Processor
from support.meta import SingletonMeta
from support.mixins import Node, Data

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Equation", "Calculation", "Calculator"]
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
    def arguments(self): return tuple([self.key, self.name, self.type])
    @property
    def parameters(self): return dict(formatter=self.formatter, style=self.style)
    def copy(self): return type(self)(*self.arguments, **self.parameters)

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
        self.__position = str(position.name).lower() if isinstance(position, Enum) else position

    def execute(self, order):
        wrapper = lambda *contents: contents[order.index(self)]
        wrapper.__name__ = "_".join([str(self.name).lower(), "independent", "variable"])
        return wrapper

    @abstractmethod
    def locate(self, *args, **kwargs): pass
    @property
    def parameters(self): return super().parameters | dict(position=self.position)
    @property
    def position(self): return self.__position


class Source(Independent):
    def __init__(self, *args, locator, **kwargs):
        assert isinstance(locator, str)
        super().__init__(*args, **kwargs)
        self.__locator = locator

    def locate(self, *args, **kwargs):
        contents = args[self.position] if isinstance(self.position, int) else kwargs[self.position]
        assert isinstance(contents, (xr.Dataset, pd.DataFrame))
        return contents[self.locator]

    @property
    def parameters(self): return super().parameters | dict(locator=self.locator)
    @property
    def locator(self): return self.__locator


class Constant(Independent):
    def locate(self, *args, **kwargs):
        content = args[self.position] if isinstance(self.position, int) else kwargs[self.position]
        assert isinstance(content, Number)
        return content


class Dependent(Variable):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(function, types.LambdaType)
        self.__function = function

    def execute(self, order):
        executes = [variable.execute(order) for variable in self.domain]
        wrapper = lambda *args, **kwargs: self.function(*[execute(*args, **kwargs) for execute in executes])
        wrapper.__name__ = "_".join([str(self.name).lower(), "dependent", "variable"])
        return wrapper

    @kwargsdispatcher("type")
    def calculate(self, *args, **kwargs): raise TypeError(kwargs["type"].__name__)

    @calculate.register.value(xr.DataArray)
    def dataarray(self, execute, sources, constants, *args, **kwargs):
        assert all([isinstance(source, xr.DataArray) for source in sources])
        assert all([isinstance(constant, Number) for constant in constants])
        contents = list(sources) + list(constants)
        dataarray = xr.apply_ufunc(execute, *contents, output_dtypes=[self.type], vectorize=True)
        dataarray = dataarray.rename(self.name)
        return dataarray

    @calculate.register.value(pd.Series)
    def series(self, execute, sources, constants, *args, **kwargs):
        assert all([isinstance(source, pd.Series) for source in sources])
        assert all([isinstance(constant, Number) for constant in constants])
        function = lambda array, *arguments: execute(*list(array), *arguments)
        series = pd.concat(list(sources), axis=1).apply(function, axis=1, raw=True, args=tuple(constants))
        series = series.rename(self.name)
        return series

    @property
    def constants(self): return [stage for stage in self.leafs if isinstance(stage, Constant)]
    @property
    def sources(self): return [stage for stage in self.leafs if isinstance(stage, Source)]
    @property
    def parameters(self): return super().parameters | dict(function=self.function)
    @property
    def domain(self): return list(self.children)
    @property
    def function(self): return self.__function


class EquationMeta(SingletonMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.items() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __iter__(cls): return iter(cls.__variables__.items())
    def __init__(cls, name, bases, attrs, *args, **kwargs):
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

        def wrapper(*args, **kwargs):
            sources = ODict([(stage, stage.locate(*args, **kwargs)) for stage in variable.sources])
            constants = ODict([(stage, stage.locate(*args, **kwargs)) for stage in variable.constants])
            order = list(sources.keys()) + list(constants.keys())
            execute = variable.execute(order)
            sources = list(sources.values())
            constants = list(constants.values())
            astype = set([type(content) for content in sources])
            assert len(astype) == 1
            results = variable.calculate(execute, sources, constants, type=list(astype)[0])
            return results

        wrapper.__name__ = "_".join([str(variable.name).lower(), "equation"])
        return wrapper

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

    def __call__(cls, *args, **kwargs):
        parameters = dict(equation=cls.__equation__)
        instance = super(CalculationMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def registry(cls): return cls.__registry__
    @property
    def fields(cls): return cls.__fields__


class Calculation(ABC, metaclass=CalculationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, equation, **kwargs): self.__equation = equation
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


class Calculator(Data, Processor, ABC, title="Calculated"):
    def __init_subclass__(cls, *args, calculation, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__calculation__ = calculation

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculation = self.__class__.__calculation__
        fields = ODict([(key, []) for key in list(calculation.fields.keys())])
        for field, calculation in iter(calculation):
            for key, value in field.items():
                fields[key].append(value)
        fields = ODict([(key, kwargs.get(key, values)) for key, values in fields.items()])
        assert all([isinstance(values, list) for values in fields.values()])
        fields = [[(key, value) for value in values] for key, values in fields.items()]
        fields = [Fields(ODict(mapping)) for mapping in product(*fields)]
        calculations = {field: calculation(*args, **kwargs) for field, calculation in iter(calculation) if field in fields}
        self.__calculations = calculations

    @property
    def calculations(self): return self.__calculations

