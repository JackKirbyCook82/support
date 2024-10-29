# -*- coding: utf-8 -*-
"""
Created on Thurs Cot 17 2024
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import pandas as pd
import xarray as xr
from copy import copy
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.dispatchers import kwargsdispatcher, typedispatcher
from support.meta import RegistryMeta
from support.mixins import SingleNode

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Equation", "Variable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


@kwargsdispatcher("datatype")
def vectorize(*args, datatype, **kwargs): raise TypeError(datatype)

@vectorize.register.value(xr.DataArray)
def vectorize_dataarray(execute, contents, constants, *args, varname, vartype, **kwargs):
    dataarray = xr.apply_ufunc(execute, *contents, *constants, output_dtypes=[vartype], vectorize=True)
    return dataarray.astype(vartype).rename(varname)

@vectorize.register.value(pd.Series)
def vectorize_series(execute, contents, constants, *args, varname, vartype, **kwargs):
    series = pd.concat(contents, axis=1).apply(execute, axis=1, raw=False, args=tuple(constants))
    return series.astype(vartype).rename(varname)

def calculate(execute, contents, constants, *args, varname, vartype, **kwargs):
    content = execute(*contents, *constants)
    return content.astype(vartype).rename(varname)


class Variable(SingleNode, ABC, metaclass=RegistryMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, **kwargs):
        if issubclass(cls, Variable) and cls is not Variable:
            return SingleNode.__new__(cls)
        function = kwargs.get("function", None)
        datatype = args[-1]
        subclass = cls[(bool(function), datatype)]
        return subclass(*args, **kwargs)

    def __init__(self, varkey, varname, vartype, datatype, *args, domain, **kwargs):
        super().__init__(*args, **kwargs)
        self.__datatype = datatype
        self.__vartype = vartype
        self.__varname = varname
        self.__varkey = varkey
        self.__domain = domain
        self.__content = None

    def __bool__(self): return self.content is not None
    def __repr__(self): return str(self.varkey)
    def __str__(self): return str(self.varname)
    def __len__(self): return int(self.size)

    def __setitem__(self, key, variable): self.set(key, variable)
    def __getitem__(self, key): return self.get(key)

    @property
    def parameters(self): return dict(varname=self.varname, vartype=self.vartype, datatype=self.datatype)
    @property
    def sources(self):
        generator = (variable for child in self.children for variable in child.sources)
        if bool(self): yield self
        else: yield from generator

    @property
    def content(self): return self.__content
    @content.setter
    def content(self, content): self.__content = content

    @property
    def datatype(self): return self.__datatype
    @property
    def vartype(self): return self.__vartype
    @property
    def varname(self): return self.__varname
    @property
    def varkey(self): return self.__varkey
    @property
    def domain(self): return self.__domain


class Dependent(Variable, register=[(True, pd.Series), (True, xr.DataArray)]):
    def __init__(self, *args, function, **kwargs):
        domain = list(inspect.signature(function).parameters.keys())
        super().__init__(*args, domain=domain, **kwargs)
        self.__vectorize = kwargs.get("vectorize", False)
        self.__function = function

    def execute(self, order):
        domains = [child.execute(order) for child in self.children]
        calculating = lambda *contents: self.function(*[domain(*contents) for domain in domains])
        sourcing = lambda *contents: contents[order.index(self)]
        wrapper = sourcing if bool(self) else calculating
        wrapper.__order__ = [str(variable) for variable in order]
        wrapper.__name__ = str(self)
        return wrapper

    def calculation(self, *args, **kwargs):
        sources = list(set(self.sources))
        contents = ODict([(variable, variable.content) for variable in sources if not isinstance(variable, Constant)])
        constants = ODict([(variable, variable.content) for variable in sources if isinstance(variable, Constant)])
        order = list(contents.keys()) + list(constants.keys())
        contents, constants = list(contents.values()), list(constants.values())
        execute = self.execute(order)
        calculation = vectorize if bool(self.vectorize) else calculate
        content = calculation(execute, contents, constants, **self.parameters)
        self.content = content
        return content

    @property
    def vectorize(self): return self.__vectorize
    @property
    def function(self): return self.__function


class NonDependent(Variable):
    def __init__(self, *args, locator, **kwargs):
        super().__init__(*args, domain=[], **kwargs)
        self.__locator = locator

    def execute(self, order):
        wrapper = lambda *contents: contents[order.index(self)]
        wrapper.__order__ = [str(variable) for variable in order]
        wrapper.__name__ = str(self)
        return wrapper

    @property
    def locator(self): return self.__locator


class Independent(NonDependent, register=[(False, pd.Series), (False, xr.DataArray)]): pass
class Constant(NonDependent, register=(False, types.NoneType)): pass


class EquationMeta(ABCMeta):
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
    def __init__(self, sources, *args, variables, **kwargs):
        for variable in variables.values():
            if isinstance(variable, Dependent): continue
            content = sources.get(variable.locator, kwargs.get(variable.locator, None))
            variable.content = content
        self.__variables = variables

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        for key in list(self.variables.keys()): del self.variables[key]

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        return variables[attribute].calculation

    def __getitem__(self, attribute):
        variables = {str(variable): variable for variable in self.variables.values()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        return variables[attribute].content

    @property
    def variables(self): return self.__variables


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
    def combine(self, content, *contents): raise TypeError(type(content))
    @combine.register(xr.DataArray)
    def dataarray(self, content, *contents): return xr.merge([content] + list(contents))
    @combine.register(pd.Series)
    def series(self, content, *contents): return pd.concat([content] + list(contents), axis=1)

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def equation(self): return self.__equation




