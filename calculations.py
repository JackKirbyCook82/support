# -*- coding: utf-8 -*-
"""
Created on Thurs Cot 17 2024
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import inspect
import pandas as pd
import xarray as xr
from copy import copy
from numbers import Number
from abc import ABC, abstractmethod
from collections import OrderedDict as ODict

from support.meta import SingletonMeta, RegistryMeta
from support.dispatchers import typedispatcher
from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Equation", "Variable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Variable(Node, ABC, metaclass=RegistryMeta):
    def __new__(cls, *args, **kwargs):
        if issubclass(cls, Variable) and cls is not Variable:
            return object.__new__(cls)
        function = kwargs.get("function", None)
        return Variable[bool(function)](*args, **kwargs)

    def __init__(self, varname, vartype, *args, domain, **kwargs):
        super().__init__(*args, **kwargs)
        self.__domain = domain
        self.__vartype = vartype
        self.__varname = varname
        self.__content = None

    def __bool__(self): return self.content is not None
    def __str__(self): return str(self.varname)
    def __len__(self): return int(self.size)

    def __setitem__(self, key, variable): self.set(key, variable)
    def __getitem__(self, key): return self.get(key)

    @property
    def sources(self):
        generator = (variable for child in self.children for variable in child.sources)
        if bool(self): yield self
        else: yield from generator

    @abstractmethod
    def execute(self, order): pass

    @property
    def content(self): return self.__content
    @content.setter
    def content(self, content): self.__content = content

    @property
    def domain(self): return self.__domain
    @property
    def varname(self): return self.__varname
    @property
    def vartype(self): return self.__vartype


class Dependent(Variable, register=True):
    def __init__(self, *args, function, **kwargs):
        domain = list(inspect.signature(function).parameters.keys())
        super().__init__(*args, domain=domain, **kwargs)
        self.__function = function

    def execute(self, order):
        domains = [child.execute(order) for child in self.children]
        wrapper = lambda *content: self.function(*[domain(*content) for domain in domains])
        wrapper.__order__ = [str(variable) for variable in order]
        wrapper.__name__ = str(self)
        return wrapper

    def calculate(self, *args, **kwargs):
        if bool(self): return self.content
        sources = ODict([(variable, variable.content) for variable in set(self.sources)])
        order, contents = list(sources.keys()), list(sources.values())
        datatype = set([type(content) for content in contents if not isinstance(content, Number)])
        assert len(datatype) == 1
        datatype = list(datatype)[0]
        execute = self.execute(order)

        ###
        ###
        ###

    @property
    def function(self): return self.__function


class Independent(Variable, register=False):
    def __init__(self, *args, locator, **kwargs):
        super().__(*args, domain=[], **kwargs)
        self.__locator = locator

    def execute(self, order):
        wrapper = lambda *contents: contents[order.index(self)]
        wrapper.__order__ = [str(variable) for variable in order]
        wrapper.__name__ = str(self)
        return wrapper

    @property
    def locator(self): return self.__locator


class EquationMeta(SingletonMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.items() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = {key: variable for key, variable in getattr(cls, "__variables__", {}).items()}
        updated = {key: variable for key, variable in attrs.items() if isinstance(variable, Variable)}
        cls.__variables__ = existing | updated

    def __call__(cls, *args, **kwargs):
        variables = {key: copy(variable) for key, variable in cls.__variables__.items()}
        for variable in variables.values():
            for key in list(variable.domain):
                variable[key] = variables[key]
        instance = super(EquationMeta, cls).__call__(*args, variables=variables, **kwargs)
        return instance


class Equation(ABC, metaclass=EquationMeta):
    def __init__(self, sources, args, variables, **kwargs):
        independents = {key: variable for key, variable in variables.items() if isinstance(variable, Independent)}
        dependents = {key: variable for key, variable in variables.items() if isinstance(variable, Dependent)}
        for variable in independents:
            variable.content = sources.get(variable.locator, kwargs.get(variable.locator, None))
        self.__independents = independents
        self.__dependents = dependents

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        independents = list(self.independents.keys())
        dependents = list(self.dependents.keys())
        for key in independents: del self.independents[key]
        for key in dependents: del self.dependents[key]

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        return variables[attribute].calculate

    def __getitem__(self, attribute):
        variables = {str(variable): variable for variable in self.variables.values()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        return variables[attribute].content

    @property
    def variables(self): return self.dependents | self.independents
    @property
    def independents(self): return self.__independents
    @property
    def dependents(self): return self.__dependents


class Calculation(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__equation__ = kwargs.get("equation", getattr(cls, "__equation__", None))

    def __init__(self, *args, **kwargs):
        self.__equation = self.__class__.__equation__

    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        contents = list(generator)
        assert all([isinstance(content, type(contents[0])) for content in contents[1:]])
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




