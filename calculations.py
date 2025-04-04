# -*- coding: utf-8 -*-
"""
Created on Thurs Cot 17 2024
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import numpy as np
import xarray as xr
import pandas as pd
from copy import copy
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import RegistryMeta, AttributeMeta
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
    def execute(self, order):
        wrapper = lambda arguments, parameters: parameters.get(self)
        wrapper.__name__ = str(self)
        return wrapper


class IndependentVariable(SourceVariable, attribute="Independent"):
    def execute(self, order):
        wrapper = lambda arguments, parameters: arguments[order.index(self)]
        wrapper.__name__ = str(self)
        return wrapper


class DependentVariable(Variable, attribute="Dependent"):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        self.__function = function

    def execute(self, order):
        domain = [variable.execute for variable in self.children.values()]
        if bool(self): wrapper = lambda contents: contents[order.index(self)]
        else: wrapper = lambda arguments, parameters: self.function(*[execute(arguments, parameters) for execute in domain])
        wrapper.__name__ = str(self)
        return wrapper

    @property
    def arguments(self): return [value for value in list(inspect.signature(self.function).parameters.keys()) if value.kind == value.POSITIONAL_ONLY]
    @property
    def parameters(self): return [value for value in list(inspect.signature(self.function).parameters.keys()) if value.kind == value.KEYWORD_ONLY]
    @property
    def domain(self): return [value for value in list(inspect.signature(self.function).parameters.keys()) if value.kind == value.POSITIONAL_OR_KEYWORD]

    @property
    def sources(self):
        children = self.children.values()
        if not bool(self): generator = (variable for child in children for variable in child.sources)
        else: generator = iter([self])
        yield from generator

    @property
    def function(self): return self.__function


class Location(object):
    def __init__(self, variable): self.variable = variable
    def __call__(self, *args, **kwargs):
        assert bool(self.variable)
        return self.variable.content


class Algorithm(ABC, metaclass=RegistryMeta):
    def __init__(self, variable): self.variable = variable
    def __call__(self, *args, **kwargs):
        sources = list(set(self.variable.sources))
        independents = ODict([(source, source.content) for source in sources if isinstance(source, IndependentVariable)])
        constants = ODict([(source, source.content) for source in sources if isinstance(source, DependentVariable)])
        assert None not in list(independents.values()) + list(constants.values())
        parameters = dict(constants.items())
        arguments = list(independents.values())
        order = list(independents.keys())
        calculation = self.variable.execute(order)
        content = self.calculate(calculation, arguments, parameters)
        content = content.astype(self.variable.vartype)
        self.variable.content = content
        return content

    @abstractmethod
    def calculate(self, *args, **kwargs): pass


class ArrayAlgorithm(Algorithm, register=ArrayVectorAlgorithm):
    def calculate(self, calculation, arguments, parameters):
        assert all([isinstance(arguments, (xr.DataArray, np.number)) for arguments in arguments])
        assert all([isinstance(parameter, np.number) for parameter in parameters.values()])
        wrapper = lambda *contents, **numbers: calculation(list(contents), dict(numbers))
        return xr.apply_ufunc(wrapper, *arguments, kwargs=parameters, output_dtypes=[self.variable.vartype], vectorize=True)


class TableAlgorithm(Algorithm, register=TableVectorAlgorithm):
    def calculate(self, calculation, arguments, parameters):
        assert all([isinstance(argument, pd.Series) for argument in arguments])
        assert all([isinstance(parameter, np.number) for parameter in parameters.values()])
        wrapper = lambda series, **numbers: calculation(list(series), dict(numbers))
        return pd.concat(arguments, axis=1).apply(wrapper, axis=1, raw=True, **parameters)


class NonVectorAlgorithm(Algorithm, register=[ArrayNonVectorAlgorithm, TableNonVectorAlgorithm]):
    def calculate(self, calculation, arguments, parameters):
        assert all([isinstance(argument, (xr.DataArray, pd.Series)) for argument in arguments])
        assert all([isinstance(parameter, np.number) for parameter in parameters.values()])
        return calculation(list(arguments), dict(parameters))


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
            for key in list(variable.domain):
                variable[key] = variables[key]
        algorithm = AlgorithmType(cls.datatype, cls.vectorize)
        parameters = dict(variables=variables, algorithm=Algorithm[algorithm], location=Location)
        return super(EquationMeta, cls).__call__(*args, **parameters, **kwargs)

    @property
    def registry(cls): return cls.__registry__
    @property
    def vectorize(cls): return cls.__vectorize__
    @property
    def datatype(cls): return cls.__datatype__


class Equation(ABC, metaclass=EquationMeta):
    def __new__(cls, sources, *args, variables, **kwargs):
        for variable in variables.values():
            if isinstance(variable, DependentVariable): continue
            elif isinstance(variable, IndependentVariable): content = sources.get(variable.locator, None)
            elif isinstance(variable, ConstantVariable): content = kwargs.get(variable.locator, None)
            else: raise TypeError(type(variable))
            if not isinstance(content, (cls.datatype, np.number)): raise ValueError(content)
            variable.content = content
        instance = super().__new__(cls)
        return instance

    def __init__(self, *args, variables, algorithm, location, **kwargs):
        self.__variables = variables
        self.__algorithm = algorithm
        self.__location = location

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        for key in list(self.variables.keys()): del self.variables[key]

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        variable = variables[attribute]
        if variable.terminal: content = self.location(variable)
        else: content = self.algorithm(variable)
        return str(variable), content

    @property
    def variables(self): return self.__variables
    @property
    def algorithm(self): return self.__algorithm
    @property
    def location(self): return self.__location


class CalculationMeta(ABCMeta, metaclass=RegistryMeta):
    def __init__(cls, *args, **kwargs):
        super(CalculationMeta, cls).__init__(*args, **kwargs)
        cls.__equation__ = kwargs.get("equation", getattr(cls, "__equation__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))

    def __call__(cls, *args, **kwargs):
        cls = type(cls.__name__, (cls, cls[cls.equation.datatype]), {})
        instance = super(CalculationMeta, cls).__call__(*args, **kwargs)
        return instance

    @property
    def equation(cls): return cls.__equation__
    @property
    def datatype(cls): return cls.__datatype__


class Calculation(ABC, metaclass=RegistryMeta):
    def __init__(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)

    def __call__(self, *args, **kwargs):
        generator = self.generator(*args, **kwargs)
        contents = dict(generator)
        return self.execute(contents, *args, **kwargs)

    @abstractmethod
    def execute(self, contents, *args, **kwargs): pass
    @abstractmethod
    def generator(self, *args, **kwargs): pass


class ArrayCalculation(Calculation, ABC, register=xr.DataArray):
    def execute(self, contents, *args, **kwargs):
        assert all([isinstance(content, (xr.DataArray, np.number)) for content in contents.values()])
        dataarrays = {name: content for name, content in contents.items() if isinstance(content, xr.DataArray)}
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        datasets = [dataarray.to_dataset(name=name) for name, dataarray in dataarrays.items()]
        dataset = xr.merge(datasets)
        for name, content in numerics.items(): dataset[name] = content
        return dataset


class TableCalculation(Calculation, ABC, register=pd.Series):
    def execute(self, contents, *args, **kwargs):
        assert all([isinstance(content, (pd.Series, np.number)) for content in contents.values()])
        series = {name: content for name, content in contents.items() if isinstance(content, pd.Series)}
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        dataframe = pd.concat(list(series.values()), axis=1)
        for name, content in numerics.items(): dataframe[name] = content
        return dataframe



