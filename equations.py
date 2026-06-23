# -*- coding: utf-8 -*-
"""
Created on Tues Apr 14 2026
@name:   Equation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import pandas as pd
from typing import Callable
from dataclasses import dataclass
from graphlib import TopologicalSorter, CycleError

from support.mixins import Mixin
from support.meta import Meta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equations"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Equation:
    variable: str; function: Callable; arguments: tuple[str, ...]; parameters: tuple[str, ...]

    @classmethod
    def create(cls, variable, function):
        signature = inspect.signature(function).parameters.items()
        arguments = [variable for variable, details in signature if details.kind in (details.POSITIONAL_OR_KEYWORD, details.POSITIONAL_OR_KEYWORD)]
        parameters = [variable for variable, details in signature if details.kind == details.KEYWORD_ONLY]
        return cls(variable, function, tuple(arguments), tuple(parameters))


class EquationError(Exception): pass
class EquationIndependentError(EquationError): pass
class EquationDependentError(EquationError): pass
class EquationConstantError(EquationError): pass
class EquationArgumentError(EquationError): pass
class EquationParameterError(EquationError): pass
class EquationOrderingError(EquationError): pass


class EquationsMeta(Meta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        criteria = lambda function: isinstance(function, types.FunctionType) and function.__name__ == "<lambda>"
        equations = [variable for variable, function in attrs.items() if criteria(function)]
        attrs = {key: value for key, value in attrs.items() if key not in equations}
        return super().__new__(mcs, name, bases, attrs, *args, **kwargs)

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super().__init__(name, bases, attrs, *args, **kwargs)
        criteria = lambda function: isinstance(function, types.FunctionType) and function.__name__ == "<lambda>"
        updated = {variable: function for variable, function in attrs.items() if criteria(function)}
        inherited = {variable: function for base in bases for variable, function in getattr(base, "__functions__", {}).items()}
        functions = inherited | updated
        updated = list(kwargs.get("variables", updated.keys()))
        inherited = [variable for base in bases for variable in getattr(base, "__variables__", [])]
        variables = list(dict.fromkeys(inherited + updated))
        updated = kwargs.get("parameters", {})
        inherited = {parameter: kwargs.get(parameter, default) for base in bases for parameter, default in getattr(base, "__parameters__", {}).items()}
        parameters = inherited | updated
        cls.__parameters__ = parameters
        cls.__functions__ = functions
        cls.__variables__ = variables

    def __call__(cls, *args, **kwargs):
        signatures = [inspect.signature(function).parameters.items() for function in cls.functions.values()]
        arguments = {variable for signature in signatures for variable, details in signature if details.kind in (details.POSITIONAL_OR_KEYWORD, details.POSITIONAL_OR_KEYWORD)}
        dependents = {argument for argument in arguments if argument in cls.functions.keys()}
        independents = {argument for argument in arguments if argument not in cls.functions.keys()}
        equations = {variable: Equation.create(variable, function) for variable, function in cls.functions.items()}
        constants = {variable: kwargs.get(variable, default) for variable, default in cls.parameters.items()}
        instance = super().__call__(*args, equations=equations, independents=independents, dependents=dependents, constants=constants, **kwargs)
        return instance

    @property
    def functions(cls): return cls.__functions__
    @property
    def variables(cls): return cls.__variables__
    @property
    def parameters(cls): return cls.__parameters__


class Equations(Mixin, metaclass=EquationsMeta):
    def __getitem__(self, equation): return self.equations[equation]
    def __init__(self, *args, equations, dependents, independents, constants, **kwargs):
        super().__init__(*args, **kwargs)
        self.__independents = independents
        self.__dependents = dependents
        self.__constants = constants
        self.__equations = equations

    def execute(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        dataframe = dataframe.copy()
        for equation in self.order:
            try: arguments = [dataframe[argument] for argument in equation.arguments]
            except KeyError: raise EquationArgumentError()
            try: parameters = {parameter: kwargs[parameter] if parameter in kwargs.keys() else self.constants[parameter] for parameter in equation.parameters}
            except KeyError: raise EquationParameterError()
            series = equation.function(*arguments, **parameters)
            dataframe[equation.variable] = series
        if not bool(type(self).variables): return dataframe
        columns = list(type(self).variables)
        return dataframe[columns]

    @property
    def order(self):
        dependency = {variable: equation.arguments for variable, equation in self.equations.items()}
        try: order = tuple(TopologicalSorter(dependency).static_order())
        except CycleError: raise EquationOrderingError()
        return [self.equations[variable] for variable in order if variable in dependency]

    @property
    def independents(self): return self.__independents
    @property
    def dependents(self): return self.__dependents
    @property
    def constants(self): return self.__constants
    @property
    def equations(self): return self.__equations


