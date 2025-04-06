# -*- coding: utf-8 -*-
"""
Created on Thurs Cot 17 2024
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import inspect
import numpy as np
import xarray as xr
import pandas as pd
from copy import copy
from itertools import chain
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import RegistryMeta, AttributeMeta
from support.decorators import ValueDispatcher
from support.trees import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Equation", "Variable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


AlgorithmType = ntuple("AlgorithmType", "datatype vectorize")
ArrayVectorAlgorithm = AlgorithmType(xr.DataArray, True)
TableVectorAlgorithm = AlgorithmType(pd.Series, True)
ArrayNonVectorAlgorithm = AlgorithmType(xr.DataArray, False)
TableNonVectorAlgorithm = AlgorithmType(pd.Series, False)


class Domain(ntuple("Domain", "arguments parameters")):
    def __iter__(self): return chain(self.arguments, self.parameters)


class Variable(Node, ABC, metaclass=AttributeMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, varkey, varname, vartype, *args, **kwargs):
        super().__init__(*args, linear=False, multiple=False, **kwargs)
        self.__vartype = vartype
        self.__varname = varname
        self.__varkey = varkey
        self.__content = None

    def __bool__(self): return self.content is not None
    def __repr__(self): return str(self.varkey)
    def __str__(self): return str(self.varname)
    def __len__(self): return int(self.size)

    @abstractmethod
    def execute(self, order): pass

    @property
    def sources(self):
        children = self.children.values()
        if not bool(self): generator = (variable for child in children for variable in child.sources)
        else: generator = iter([self])
        yield from generator

    @property
    def vartype(self): return self.__vartype
    @property
    def varname(self): return self.__varname
    @property
    def varkey(self): return self.__varkey

    @property
    def content(self): return self.__content
    @content.setter
    def content(self, content): self.__content = content


class SourceVariable(Variable, ABC):
    def __init__(self, *args, locator, **kwargs):
        super().__init__(*args, **kwargs)
        self.__locator = locator

    @property
    def locator(self): return self.__locator

class ConstantVariable(SourceVariable, attribute="Constant"):
    def execute(self, order): return lambda arguments, parameters: parameters.get(str(self))

class IndependentVariable(SourceVariable, attribute="Independent"):
    def execute(self, order): return lambda arguments, parameters: arguments[order.index(self)]


class DependentVariable(Variable, attribute="Dependent"):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        signature = inspect.signature(function).parameters.items()
        arguments = [key for key, value in signature if value.kind != value.KEYWORD_ONLY]
        parameters = [key for key, value in signature if value.kind == value.KEYWORD_ONLY]
        domain = Domain(arguments, parameters)
        self.__function = function
        self.__domain = domain

    def execute(self, order):
        children = list(self.children.items())
        if bool(self): return lambda arguments, parameters: arguments[order.index(self)]
        primary = [variable.execute(order) for key, variable in children if key in self.domain.arguments]
        secondary = {key: variable.execute(order) for key, variable in children if key in self.domain.parameters}
        executes = Domain(primary, secondary)
        primary = lambda arguments, parameters: [execute(arguments, parameters) for execute in executes.arguments]
        secondary = lambda arguments, parameters: {key: execute(arguments, parameters) for key, execute in executes.parameters.items()}
        return lambda arguments, parameters: self.function(*primary(arguments, parameters), **secondary(arguments, parameters))

    @property
    def function(self): return self.__function
    @property
    def domain(self): return self.__domain


class Algorithm(ABC, metaclass=RegistryMeta):
    def __init__(self, variable): self.variable = variable
    def __call__(self, *args, **kwargs):
        sources = list(set(self.variable.sources))
        arguments = ODict([(source, source.content) for source in sources if isinstance(source, IndependentVariable)])
        parameters = ODict([(source, source.content) for source in sources if isinstance(source, ConstantVariable)])
        parameters = {str(variable): content for variable, content in parameters.items()}
        order = list(arguments.keys())
        arguments = list(arguments.values())
        calculation = self.variable.execute(order)
        name = str(self.variable)
        content = self.calculate(calculation, arguments, parameters)
        content = content.astype(self.variable.vartype)
        self.variable.content = content
        return name, content

    @abstractmethod
    def calculate(self, *args, **kwargs): pass


class ArrayAlgorithm(Algorithm, register=ArrayVectorAlgorithm):
    def calculate(self, calculation, arguments, parameters):
        assert all([isinstance(argument, (xr.DataArray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters])
        function = lambda *dataarrays, **constants: calculation(dataarrays, constants)
        return xr.apply_ufunc(function, *arguments, kwargs=parameters, output_dtypes=[self.variable.vartype], vectorize=True)

class TableAlgorithm(Algorithm, register=TableVectorAlgorithm):
    def calculate(self, calculation, arguments, parameters):
        assert all([isinstance(argument, pd.Series) for argument in arguments])
        assert not any([isinstance(parameter, pd.Series) for parameter in parameters])
        function = lambda dataframe, **constants: calculation(dataframe, constants)
        return pd.concat(arguments, axis=1).apply(function, axis=1, raw=True, **parameters)

class NonVectorAlgorithm(Algorithm, register=[ArrayNonVectorAlgorithm, TableNonVectorAlgorithm]):
    def calculate(self, calculation, arguments, parameters):
        assert all([isinstance(argument, (xr.DataArray, pd.Series)) for argument in arguments])
        assert not any([isinstance(parameter, (xr.DataArray, pd.Series)) for parameter in parameters])
        return calculation(arguments, parameters)


class EquationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.items() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = {key: variable for key, variable in getattr(cls, "__registry__", {}).items()}
        updated = {key: variable for key, variable in attrs.items() if isinstance(variable, Variable)}
        cls.__vectorize__ = kwargs.get("vectorize", getattr(cls, "__vectorize__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__registry__ = dict(existing) | dict(updated)

    def __call__(cls, *args, **kwargs):
        variables = {key: copy(variable) for key, variable in cls.registry.items()}
        for variable in variables.values():
            if isinstance(variable, SourceVariable): continue
            for key in list(variable.domain):
                variable[key] = variables[key]
        algorithm = AlgorithmType(cls.datatype, cls.vectorize)
        parameters = dict(variables=variables, algorithm=Algorithm[algorithm])
        return super(EquationMeta, cls).__call__(*args, **parameters, **kwargs)

    @property
    def registry(cls): return cls.__registry__
    @property
    def vectorize(cls): return cls.__vectorize__
    @property
    def datatype(cls): return cls.__datatype__


class Equation(ABC, metaclass=EquationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, sources, *args, variables, **kwargs):
        for variable in variables.values():
            if isinstance(variable, DependentVariable): continue
            elif isinstance(variable, IndependentVariable): content = sources.get(variable.locator, None)
            elif isinstance(variable, ConstantVariable): content = kwargs.get(variable.locator, None)
            else: raise TypeError(type(variable))
            variable.content = content
        instance = super().__new__(cls)
        return instance

    def __init__(self, *args, variables, algorithm, **kwargs):
        self.__variables = variables
        self.__algorithm = algorithm

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        for key in list(self.variables.keys()): del self.variables[key]

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        variable = variables[attribute]
        if variable.terminal: return lambda: (str(variable), variable.content)
        else: return self.algorithm(variable)

    @property
    def variables(self): return self.__variables
    @property
    def algorithm(self): return self.__algorithm


class Calculation(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__equation__ = kwargs.get("equation", getattr(cls, "__equation__", None))

    def __init__(self, *args, **kwargs): assert inspect.isgeneratorfunction(self.execute)
    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        method = self.equation.datatype
        contents = dict(generator)
        content = self.combine(contents, *args, method=method, **kwargs)
        return content

    @ValueDispatcher(locator="method")
    def combine(self, contents, *args, method, **kwargs): pass

    @combine.register(xr.DataArray)
    def dataarray(self, contents, *args, **kwargs):
        assert all([isinstance(content, (xr.DataArray, np.number)) for content in contents.values()])
        dataarrays = {name: content for name, content in contents.items() if isinstance(content, xr.DataArray)}
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        datasets = [dataarray.to_dataset(name=name) for name, dataarray in dataarrays.items()]
        dataset = xr.merge(datasets)
        for name, content in numerics.items(): dataset[name] = content
        return dataset

    @combine.register(pd.Series)
    def series(self, contents, *args, **kwargs):
        assert all([isinstance(content, (pd.Series, np.number)) for content in contents.values()])
        series = [content.rename(name) for name, content in contents.items() if isinstance(content, pd.Series)]
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        dataframe = pd.concat(list(series), axis=1)
        for name, content in numerics.items(): dataframe[name] = content
        return dataframe

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def equation(self): return type(self).__equation__


