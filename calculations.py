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
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.decorators import TypeDispatcher
from support.trees import NonLinearSingleNode
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Equation", "Variable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


AlgorithmType = ntuple("AlgorithmType", "vectorized datatype")
VariableType = ntuple("VariableType", "calculated datatype")
NonVectorizedTable = AlgorithmType(False, pd.Series)
NonVectorizedArray = AlgorithmType(False, xr.DataArray)
VectorizedTable = AlgorithmType(True, pd.Series)
VectorizedArray = AlgorithmType(True, xr.DataArray)
NonCalculatedConstant = VariableType(False, types.NoneType)
NonCalculatedTable = VariableType(False, pd.Series)
NonCalculatedArray = VariableType(False, xr.DataArray)
CalculatedTable = VariableType(True, pd.Series)
CalculatedArray = VariableType(True, xr.DataArray)


class Algorithm(object, metaclass=RegistryMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, execute, arguments, parameters):
        self.__parameters = parameters
        self.__arguments = arguments
        self.__execute = execute

    @property
    def parameters(self): return self.__parameters
    @property
    def arguments(self): return self.__arguments
    @property
    def execute(self): return self.__execute


class VectorizedAlgorithm(Algorithm): pass
class UnVectorizedAlgorithm(Algorithm):
    def __call__(self, *args, **kwargs):
        return self.execute(list(self.arguments), dict(self.parameters))

class TableUnVectorizedAlgorithm(UnVectorizedAlgorithm, register=NonVectorizedTable): pass
class ArrayUnVectorizedAlgorithm(UnVectorizedAlgorithm, register=NonVectorizedArray): pass

class TableVectorizedAlgorithm(VectorizedAlgorithm, register=VectorizedTable):
    def __call__(self, *args, **kwargs):
        wrapper = lambda arguments, **parameters: self.execute(list(arguments), dict(parameters))
        return pd.concat(self.arguments, axis=1).apply(wrapper, axis=1, raw=True, **self.parameters)

class ArrayVectorizedAlgorithm(VectorizedAlgorithm, register=VectorizedArray):
    def __call__(self, *args, vartype, **kwargs):
        wrapper = lambda *arguments, **parameters: self.execute(list(arguments), dict(parameters))
        return xr.apply_ufunc(wrapper, *self.arguments, output_dtypes=[vartype], vectorize=True, kwargs=self.parameters)


class Variable(NonLinearSingleNode, ABC, metaclass=RegistryMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, **kwargs):
        if issubclass(cls, Variable) and cls is not Variable:
            return NonLinearSingleNode.__new__(cls)
        function = kwargs.get("function", None)
        datatype = args[-1]
        vartype = VariableType(bool(function), datatype)
        subclass = cls[vartype]
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

    @property
    def sources(self):
        children = self.children.values()
        generator = (variable for child in children for variable in child.sources)
        if bool(self): yield self
        else: yield from generator

    @abstractmethod
    def execute(self, order): pass

    @property
    def content(self): return self.__content
    @content.setter
    def content(self, content):
        if self.datatype in [types.NoneType]: self.__content = content
        else: self.__content = content.rename(self.varname)

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


class Dependent(Variable, register=[CalculatedTable, CalculatedArray]):
    def __init__(self, *args, function, **kwargs):
        domain = list(inspect.signature(function).parameters.keys())
        super().__init__(*args, domain=domain, **kwargs)
        self.__vectorize = kwargs.get("vectorize", False)
        self.__function = function

    def __call__(self, *args, **kwargs):
        if bool(self): return self.content
        content = self.calculate(*args, **kwargs)
        self.content = content
        return self.content

    def calculate(self, *args, **kwargs):
        variables = list(set(self.sources))
        independents = ODict([(variable, variable.content) for variable in variables if not isinstance(variable, Constant)])
        constants = ODict([(variable, variable.content) for variable in variables if isinstance(variable, Constant)])
        parameters = {str(variable): content for variable, content in constants.items()}
        arguments = list(independents.values())
        order = list(independents.keys())
        execute = self.execute(order)
        algorithm = (self.vectorize, self.datatype)
        algorithm = Algorithm[algorithm](execute, arguments, parameters)
        return algorithm(*args, vartype=self.vartype, **kwargs)

    def execute(self, order):
        execution = [child.execute(order) for child in self.children]
        source = lambda arguments, parameters: parameters[str(self)] if str(self) in parameters else arguments[order.index(self)]
        calculate = lambda arguments, parameters: self.function(*[execute(arguments, parameters) for execute in execution])
        wrapper = source if bool(self) else calculate
        wrapper.__name__ = str(self)
        return wrapper

    @property
    def vectorize(self): return self.__vectorize
    @property
    def function(self): return self.__function


class Source(Variable):
    def __init__(self, *args, locator, **kwargs):
        super().__init__(*args, domain=[], **kwargs)
        self.__locator = locator

    def execute(self, order):
        wrapper = lambda arguments, parameters: parameters[str(self)] if str(self) in parameters else arguments[order.index(self)]
        wrapper.__name__ = str(self)
        return wrapper

    @property
    def locator(self): return self.__locator


class Independent(Source, register=[NonCalculatedTable, NonCalculatedArray]): pass
class Constant(Source, register=NonCalculatedConstant): pass


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
        return variables[attribute]

    def __getitem__(self, attribute):
        variables = {str(variable): variable for variable in self.variables.values()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        variable = variables[attribute]
        if not bool(variable):
            raise ValueError(attribute)
        return variable.content

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

    @TypeDispatcher(locator=0)
    def combine(self, content, *contents): raise TypeError(type(content))
    @combine.register(xr.DataArray)
    def dataarray(self, content, *contents): return xr.merge([content] + list(contents))
    @combine.register(pd.Series)
    def series(self, content, *contents): return pd.concat([content] + list(contents), axis=1)

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def equation(self): return self.__equation




