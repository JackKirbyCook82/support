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
        inherited = {parameter for base in bases for parameter, default in getattr(base, "__hyperparams__", {}).items()}
        hyperparams = inherited | updated
        cls.__hyperparams__ = hyperparams
        cls.__functions__ = functions
        cls.__variables__ = variables

    def __call__(cls, *args, **kwargs):
        signatures = [inspect.signature(function).parameters.items() for function in cls.functions.values()]
        arguments = [variable for signature in signatures for variable, details in signature if details.kind in (details.POSITIONAL_OR_KEYWORD, details.POSITIONAL_OR_KEYWORD)]
        dependents = {argument for argument in arguments if argument in cls.functions.keys()}
        independents = {argument for argument in arguments if argument not in cls.functions.keys()}
        equations = {variable: Equation.create(variable, function) for variable, function in cls.functions.items()}
        parameters = {variable: kwargs.get(variable, default) for variable, default in cls.hyperparams.items()}
        instance = super().__call__(*args, equations=equations, independents=independents, dependents=dependents, parameters=parameters, **kwargs)
        return instance

    @property
    def functions(cls): return cls.__functions__
    @property
    def variables(cls): return cls.__variables__
    @property
    def hyperparams(cls): return cls.__hyperparams__


class Equations(Mixin, metaclass=EquationsMeta):
    def __getitem__(self, equation): return self.equations[equation]
    def __init__(self, *args, equations, dependents, independents, parameters, **kwargs):
        super().__init__(*args, **kwargs)
        self.__parameters = parameters
        self.__independents = independents
        self.__dependents = dependents
        self.__equations = equations

    def execute(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        missing = {argument for argument in self.independents if argument not in dataframe.columns and argument not in kwargs.keys()}
        if bool(missing): raise EquationArgumentError()
        dataframe = dataframe.copy()
        for equation in self.order:
            arguments = [dataframe[argument] for argument in equation.arguments]
            dataframe[equation.variable] = equation.function(*arguments, **self.parameters)
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
    def equations(self): return self.__equations
    @property
    def parameters(self): return self.__parameters


