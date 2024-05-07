# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from itertools import product
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher
from support.pipelines import Processor
from support.meta import SingletonMeta
from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Equation", "Calculation", "Calculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Fields(frozenset):
    def __getitem__(self, key): return self.todict()[key]
    def __new__(cls, mapping):
        assert isinstance(mapping, ODict)
        mapping = list(mapping.items())
        return super().__new__(cls, mapping)

    def todict(self): return ODict(list(self))
    def tolist(self): return list(self)
    def keys(self): return self.todict().keys()
    def values(self): return self.todict().values()
    def items(self): return self.todict().items()


class Variable(Node, ABC):
    def __new__(cls, *args, function=None, **kwargs):
        if cls is Variable and bool(function):
            return Dependent(*args, function=function, **kwargs)
        elif cls is Variable and not bool(function):
            return Independent(*args, **kwargs)
        return super().__new__(cls)

    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)
    def __len__(self): return self.size

    @property
    @abstractmethod
    def execute(self): pass
    @abstractmethod
    def copy(self): pass


class Independent(Variable):
    def __init__(self, name, *args, locator, **kwargs):
        assert isinstance(locator, (int, str, Enum))
        super().__init__(*args, name=name, **kwargs)
        self.locator = str(locator.name).lower() if isinstance(locator, Enum) else locator

    def copy(self):
        parameters = dict(name=self.name, formatter=self.formatter, style=self.style)
        return type(self)(locator=self.locator, **parameters)

    @property
    def execute(self):
        wrapper = lambda *args, **kwargs: args[self.locator] if isinstance(self.locator, int) else kwargs.get(self.locator, None)
        wrapper.__name__ = str(self.name).lower()
        return wrapper


class Dependent(Variable):
    def __init__(self, name, *args, function, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        assert isinstance(function, types.LambdaType)
        self.function = function

    def copy(self):
        parameters = dict(name=self.name, formatter=self.formatter, style=self.style)
        return type(self)(funciton=self.function, **parameters)

    @property
    def execute(self):
        sources = [variable.execute for variable in self.children]
        wrapper = lambda *args, **kwargs: self.function(*[source(*args, **kwargs) for source in sources])
        wrapper.__name__ = str(self.name).lower()
        return wrapper


class EquationMeta(SingletonMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.itmes() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __iter__(cls): return iter(cls.__variables__.items())
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        existing = {key: variable.copy() for key, variable in getattr(cls, "__variables__", {}).items()}
        updated = {key: variable for key, variable in attrs.items() if isinstance(variable, Variable)}
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
    def __getattr__(self, attr): return super().__getattr__(attr) if attr in self else self.variables[attr]
    def __contains__(self, attr): return attr in self.variables.keys()
    def __init__(self, *args, variables, **kwargs): self.variables = variables


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        if bool(cls):
            fields = Fields(cls.fields)
            cls.registry[fields] = cls
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is CalculationMeta for base in bases]):
            return
        if not any([type(subbase) is CalculationMeta for base in bases for subbase in base]):
            assert all([attr not in kwargs["fields"] for attr in ("equation", "domain", "fields")])
            cls.__fields__ = ODict.fromkeys(list(set(kwargs["fields"])))
            cls.__registry__ = ODict()
        equation = kwargs.get("equation", getattr(cls, "__equation__", None))
        fields = ODict([(key, kwargs.get(key, value)) for key, value in getattr(cls, "__fields__", {}).items()])
        assert issubclass(equation, Equation) if equation is not None else True
        cls.__equation__ = equation
        cls.__fields__ = fields

    def __bool__(cls): return None not in cls.fields.values()
    def __iter__(cls): return iter(list(cls.registry.items()))

    def __call__(cls, *args, **kwargs):
        parameters = dict(equation=cls.__equation__)
        instance = super(CalculationMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def registry(cls): return cls.__registry__
    @property
    def fields(cls): return cls.__fields__


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, *args, equation, **kwargs):
        self.__equation = equation

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


class Calculator(Processor, ABC, title="Calculated"):
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
        fields = ODict([(key, kwargs.get(key, values)) for key, values in field.items()])
        assert all([isinstance(values, list) for values in fields.values()])
        fields = [[(key, value) for value in values] for key, values in fields.items()]
        fields = [Fields(mapping) for mapping in product(*fields)]
        calculations = ODict([(field, calculation) for field, calculation in iter(calculation) if field in fields])
        self.__calculations = calculations

    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)

    @property
    def calculations(self): return self.__calculations

