# -*- coding: utf-8 -*-
"""
Created on Tues Apr 14 2026
@name:   Equation Objects
@author: Jack Kirby Cook
@file:   support/equations.py

"""

import inspect
import pandas as pd

from support.mixins import Mixin
from support.meta import Meta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class EquationError(Exception): pass
class EquationParameterError(EquationError): pass
class EquationFunctionError(EquationError): pass
class EquationCircularError(EquationError): pass


class EquationMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super().__init__(name, bases, attrs)
        equations = dict()
        for base in reversed(cls.__mro__):
            if not isinstance(base, EquationMeta): continue
            if not any(isinstance(value, EquationMeta) for value in base.__mro__[1:]): continue
            for name, value in vars(base).items():
                if name.startswith("_"): continue
                if callable(value):
                    signature = inspect.signature(value)
                    for parameter in signature.parameters.values():
                        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY): raise EquationParameterError()
                    equations[name] = value
                else: equations.pop(name, None)
        cls.__equations__ = equations

    def __call__(cls, *args, **kwargs):
        functions = dict(cls.equations)
        parameters = {name for function in functions.values() for name in inspect.signature(function).parameters.keys()}
        constants = {name: kwargs[name] for name in parameters if name in kwargs.keys()}
        instance = super().__call__(*args, functions=functions, constants=constants, **kwargs)
        return instance

    @property
    def equations(cls): return cls.__equations__


class Equation(Mixin, metaclass=EquationMeta):
    def __init__(self, *args, functions, constants, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(functions, dict) and isinstance(constants, dict)
        self.__functions = functions
        self.__constants = constants

    def evaluate(self, *dependents, **independents):
        domain = independents | self.constants
        cache = dict()
        order = list()

        def resolve(name):
            if name in domain: return domain[name]
            if name in cache: return cache[name]
            if name in order: raise EquationCircularError()
            try: function = self.functions[name]
            except KeyError: raise EquationFunctionError()
            order.append(name)
            try:
                keywords = dict()
                for parameter in inspect.signature(function).parameters.values():
                    dependency = parameter.name
                    if dependency in domain or dependency in self.functions or parameter.default is inspect.Parameter.empty:
                        keywords[dependency] = resolve(dependency)
                result = function(**keywords)
                cache[name] = result
                return result
            finally: order.pop()
        return {dependent: resolve(dependent) for dependent in dependents}

    @property
    def functions(self): return self.__functions
    @property
    def constants(self): return self.__constants


class DataframeEquation(Equation):
    def execute(self, independents, *dependents, **constants):
        assert isinstance(independents, pd.DataFrame)
        result = pd.DataFrame(index=independents.index)
        if not dependents: dependents = tuple(self.functions)
        dependencies = {parameter.name for function in self.functions.values() for parameter in inspect.signature(function).parameters.values()}
        dependencies = dependencies - self.functions.keys()
        series = {name: independents[name] for name in dependencies if name in independents.columns}
        numerics = {name: constants[name] for name in dependencies if name in constants}
        independents = series | numerics
        results = self.evaluate(*dependents, **independents)
        for name, series in results.items(): result[name] = series
        return result



