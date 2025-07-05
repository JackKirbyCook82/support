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
from functools import reduce
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.decorators import TypeDispatcher
from support.meta import RegistryMeta, AttributeMeta
from support.trees import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Equation", "Variable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


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
    def __init__(self, *args, locator, **kwargs):
        assert isinstance(locator, str)
        super().__init__(*args, locator=locator, **kwargs)

    def execute(self, order):
        wrapper = lambda arguments, parameters: parameters.get(str(self))
        wrapper.__name__ = repr(self)
        return wrapper

class IndependentVariable(SourceVariable, attribute="Independent"):
    def __init__(self, *args, locator, **kwargs):
        assert isinstance(locator, (str, tuple))
        locator = list(locator) if isinstance(locator, tuple) else [locator]
        super().__init__(*args, locator=locator, **kwargs)

    def execute(self, order):
        wrapper = lambda arguments, parameters: arguments[order.index(self)]
        wrapper.__name__ = repr(self)
        return wrapper


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
        if bool(self): return lambda arguments, parameters: self.content
        primary = [variable.execute(order) for key, variable in children if key in self.domain.arguments]
        secondary = {key: variable.execute(order) for key, variable in children if key in self.domain.parameters}
        executes = Domain(primary, secondary)
        primary = lambda arguments, parameters: [execute(arguments, parameters) for execute in executes.arguments]
        secondary = lambda arguments, parameters: {key: execute(arguments, parameters) for key, execute in executes.parameters.items()}
        wrapper = lambda arguments, parameters: self.function(*primary(arguments, parameters), **secondary(arguments, parameters))
        wrapper.__name__ = repr(self)
        return wrapper

    @property
    def function(self): return self.__function
    @property
    def domain(self): return self.__domain


class AlgorithmType(ntuple("AlgorithmType", "datatype vectorize")): pass
class Algorithm(ntuple("Algorithm", "variable"), ABC, metaclass=RegistryMeta):
    def __call__(self, *args, **kwargs):
        sources = list(set(self.variable.sources))
        arguments = ODict([(source, source.content) for source in sources if isinstance(source, IndependentVariable)])
        parameters = ODict([(source, source.content) for source in sources if isinstance(source, ConstantVariable)])
        parameters = {str(variable): content for variable, content in parameters.items()}
        order = list(arguments.keys())
        arguments = list(arguments.values())
        calculation = self.variable.execute(order)
        content = self.execute(calculation, arguments, parameters, *args, **kwargs)
        content = content.astype(self.variable.vartype)
        self.variable.content = content
        return self.variable.varname, content

    @abstractmethod
    def execute(self, calculation, arguments, parameters): pass


class DirectAlgorithm(Algorithm, register=[AlgorithmType(xr.DataArray, False), AlgorithmType(pd.Series, False)]):
    def execute(self, calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, (xr.DataArray, pd.Series)) for argument in arguments])
        assert not any([isinstance(parameter, (xr.DataArray, pd.Series)) for parameter in parameters])
        return calculation(arguments, parameters)

class ArrayAlgorithm(Algorithm, register=AlgorithmType(xr.DataArray, True)):
    def execute(self, calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, (xr.DataArray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters])
        function = lambda *dataarrays, **constants: calculation(dataarrays, constants)
        return xr.apply_ufunc(function, *arguments, kwargs=parameters, output_dtypes=[self.variable.vartype], vectorize=True)

class TableAlgorithm(Algorithm, register=AlgorithmType(pd.Series, True)):
    def execute(self, calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, pd.Series) for argument in arguments])
        assert not any([isinstance(parameter, pd.Series) for parameter in parameters])
        function = lambda dataframe, **constants: calculation(dataframe, constants)
        return pd.concat(arguments, axis=1).apply(function, axis=1, raw=True, **parameters)


class EquationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.items() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = [dict(base.contents) for base in bases if issubclass(type(base), EquationMeta)]
        existing = reduce(lambda lead, lag: lead | lag, existing, dict())
        updated = {key: variable for key, variable in attrs.items() if isinstance(variable, Variable)}
        cls.__vectorize__ = kwargs.get("vectorize", getattr(cls, "__vectorize__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__contents__ = dict(existing) | dict(updated)

    def __call__(cls, sources, *args, **kwargs):
        assert isinstance(sources, (list, dict, xr.Dataset, pd.DataFrame))
        assert all([isinstance(value, (xr.Dataset, pd.DataFrame)) for value in sources]) if isinstance(sources, list) else True
        assert all([isinstance(value, (xr.Dataset, pd.DataFrame)) for value in sources.values()]) if isinstance(sources, dict) else True
        variables = {key: copy(variable) for key, variable in cls.contents.items()}
        for variable in variables.values():
            if isinstance(variable, SourceVariable): continue
            for key in list(variable.domain):
                variable[key] = variables[key]
        for variable in variables.values():
            try:
                if isinstance(variable, DependentVariable): continue
                elif isinstance(variable, IndependentVariable): content = cls.locate(sources, *variable.locator)
                elif isinstance(variable, ConstantVariable): content = kwargs.get(variable.locator, None)
                else: raise TypeError(type(variable))
            except (KeyError, IndexError): content = None
            variable.content = content
        parameters = dict(variables=variables, datatype=cls.datatype, vectorize=cls.vectorize)
        return super(EquationMeta, cls).__call__(*args, **parameters, **kwargs)

    @TypeDispatcher(locator=0)
    def locate(cls, sources, locator, *locators): raise TypeError(type(sources))
    @locate.register(dict, list)
    def mapping(cls, sources, locator, *locators): return cls.locate(sources[locator], *locators)
    @locate.register(xr.Dataset, pd.DataFrame)
    def table(cls, sources, locator, *locators): return sources.get(locator, None)

    @property
    def independents(cls): return [content for content in cls.contents.values() if isinstance(content, IndependentVariable)]
    @property
    def dependents(cls): return [content for content in cls.contents.values() if isinstance(content, DependentVariable)]
    @property
    def constants(cls): return [content for content in cls.contents.values() if isinstance(content, ConstantVariable)]

    @property
    def domain(cls): return {str(content): content.locator for content in cls.independents}
    @property
    def parameters(cls): return [str(content) for content in cls.constants]
    @property
    def range(cls): return [str(content) for content in cls.dependents]

    @property
    def vectorize(cls): return cls.__vectorize__
    @property
    def datatype(cls): return cls.__datatype__
    @property
    def contents(cls): return cls.__contents__


class Equation(ABC, metaclass=EquationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, variables, datatype, vectorize, **kwargs):
        self.__variables = variables
        self.__datatype = datatype
        self.__vectorize = vectorize

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        for key in list(self.variables.keys()): del self.variables[key]

    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        yield from generator

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        variable = variables[attribute]
        if variable.terminal: return lambda: (variable.varname, variable.content)
        algorithmtype = AlgorithmType(self.datatype, self.vectorize)
        return Algorithm[algorithmtype](variable)

    def execute(self, *args, **kwargs):
        return
        yield

    @property
    def variables(self): return self.__variables
    @property
    def datatype(self): return self.__datatype
    @property
    def vectorize(self): return self.__vectorize


class Calculation(ABC, metaclass=RegistryMeta):
    def __init__(self, *args, required=[], optional=[], **kwargs):
        assert isinstance(required, list) or issubclass(required, Equation)
        assert isinstance(optional, list) or issubclass(optional, Equation)
        assert all([issubclass(equation, Equation) for equation in required]) if isinstance(required, list) else True
        assert all([issubclass(equation, Equation) for equation in optional]) if isinstance(optional, list) else True
        self.__required = list(required) if isinstance(required, list) else [required]
        self.__optional = list(optional) if isinstance(optional, list) else [optional]

    def __call__(self, sources, *args, **kwargs):
        assert isinstance(sources, (list, dict, xr.Dataset, pd.DataFrame))
        assert all([isinstance(value, (xr.Dataset, pd.DataFrame)) for value in sources]) if isinstance(sources, list) else True
        assert all([isinstance(value, (xr.Dataset, pd.DataFrame)) for value in sources.values()]) if isinstance(sources, dict) else True
        generator = self.generator(sources, *args, **kwargs)
        contents = dict(generator)
        content = self.execute(contents, *args, **kwargs)
        return content

    def generator(self, sources, *args, **kwargs):
        required = [equation for equation in self.required if all([self.locatable(sources, *locator) for locator in equation.domain.values()])]
        optional = [equation for equation in self.optional if all([self.locatable(sources, *locator) for locator in equation.domain.values()])]
        assert len(required) == len(self.required)
        equations = reversed(required + optional)
        equation = EquationMeta("Equation", tuple(equations), dict())
        with equation(sources, *args, **kwargs) as execute:
            yield from execute(*args, **kwargs)

    @TypeDispatcher(locator=0)
    def locatable(self, sources, locator, *locators): raise TypeError(type(sources))
    @locatable.register(dict, list)
    def mapping(self, sources, locator, *locators): return self.locatable(sources[locator], *locators)
    @locatable.register(xr.Dataset)
    def dataset(self, sources, locator, *locators): return locator in sources.keys()
    @locatable.register(pd.DataFrame)
    def dataframe(self, sources, locator, *locators): return locator in sources.columns

    @staticmethod
    @abstractmethod
    def execute(contents, *args, **kwargs): pass

    @property
    def required(self): return self.__required
    @property
    def optional(self): return self.__optional


class ArrayCalculation(Calculation, register=xr.DataArray):
    @staticmethod
    def execute(contents, *args, **kwargs):
        assert all([isinstance(content, (xr.DataArray, np.number)) for content in contents.values()])
        dataarrays = {name: content for name, content in contents.items() if isinstance(content, xr.DataArray)}
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        datasets = [dataarray.to_dataset(name=name) for name, dataarray in dataarrays.items()]
        dataset = xr.merge(datasets)
        for name, content in numerics.items(): dataset[name] = content
        return dataset


class TableCalculation(Calculation, register=pd.Series):
    @staticmethod
    def execute(contents, *args, **kwargs):
        assert all([isinstance(content, (pd.Series, np.number)) for content in contents.values()])
        series = [content.rename(name) for name, content in contents.items() if isinstance(content, pd.Series)]
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        dataframe = pd.concat(list(series), axis=1)
        for name, content in numerics.items(): dataframe[name] = content
        return dataframe



