# -*- coding: utf-8 -*-
"""
Created on Weds 25 2026
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import pandas as pd
from dataclasses import dataclass
from graphlib import TopologicalSorter, CycleError

from support.mixins import Mixin
from support.meta import Meta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Equation:
    variable: str; function: callable; arguments: tuple[str, ...]; parameters: tuple[str, ...]

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


class CalculationMeta(Meta):
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
        cls.__functions__ = functions
        cls.__variables__ = variables

    def __call__(cls, *args, **kwargs):
        equations = {variable: Equation.create(variable, function) for variable, function in cls.functions.items()}
        instance = super().__call__(*args, equations=equations, **kwargs)
        return instance

    @property
    def functions(cls): return cls.__functions__
    @property
    def variables(cls): return cls.__variables__


class Calculation(Mixin, metaclass=CalculationMeta):
    def __init__(self, *args, equations, **kwargs):
        super().__init__(*args, **kwargs)
        self.__equations = equations

    def calculate(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        missing = {argument for argument in self.arguments if argument not in dataframe.columns}
        if bool(missing): raise EquationArgumentError()
        missing = {parameter for parameter in self.parameters if parameter not in kwargs.keys()}
        if bool(missing): raise EquationParameterError()
        dataframe = dataframe.copy()
        for equation in self.order:
            arguments = [dataframe[argument] for argument in equation.arguments]
            parameters = {parameter: kwargs[parameter] for parameter in equation.parameters}
            dataframe[equation.variable] = equation.function(*arguments, **parameters)
        if not bool(type(self).variables): return dataframe
        columns = list(type(self).variables)
        return dataframe[columns]

    @property
    def arguments(self):
        dependency = {variable: equation.arguments for variable, equation in self.equations.items()}
        return {argument for arguments in dependency.values() for argument in arguments if argument not in dependency}

    @property
    def parameters(self):
        dependency = {variable: equation.parameters for variable, equation in self.equations.items()}
        return {parameter for parameters in dependency.values() for parameter in parameters}

    @property
    def order(self):
        dependency = {variable: equation.arguments for variable, equation in self.equations.items()}
        try: order = tuple(TopologicalSorter(dependency).static_order())
        except CycleError: raise EquationOrderingError()
        return [self.equations[variable] for variable in order if variable in dependency]

    @property
    def equations(self): return self.__equations



