# -*- coding: utf-8 -*-
"""
Created on Tues Apr 14 2026
@name:   Equation Objects
@author: Jack Kirby Cook
@file:   support/equations.py

"""

import inspect
import pandas as pd
from typing import Callable
from dataclasses import dataclass, field

from support.mixins import Mixin
from support.meta import Meta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation", "DataframeEquation"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class EquationError(Exception): pass
class EquationParameterError(EquationError): pass
class EquationFunctionError(EquationError): pass
class EquationCircularError(EquationError): pass


@dataclass(frozen=True)
class EquationNamespace:
    variables: list = field(default_factory=list)
    defaults: dict = field(default_factory=dict)
    equations: dict[str, Callable] = field(default_factory=dict)

    def __or__(self, other):
        if not isinstance(other, EquationNamespace): return TypeError(type(other))
        variables = list(dict.fromkeys(self.variables + other.variables))
        defaults = self.defaults | other.defaults
        equations = self.equations | other.equations
        return EquationNamespace(variables, defaults, equations)


class EquationMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super().__init__(name, bases, attrs)
        variables = list(kwargs.get("variables", []))
        defaults = dict(kwargs.get("defaults", {}))
        equations = dict()
        for key, value in attrs.items():
            if key.startswith("__"): continue
            if inspect.isfunction(value) and value.__name__ == "<lambda>":
                signature = inspect.signature(value)
                for parameter in signature.parameters.values():
                    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    if parameter.kind not in kinds: raise EquationParameterError()
                equations[key] = value
        localized = EquationNamespace(variables, defaults, equations)
        globalized = EquationNamespace()
        for base in reversed(cls.__mro__):
            if not isinstance(base, EquationMeta): continue
            if base.localized is None: continue
            for key, value in vars(base).items():
                if not key.startswith("_") and key not in localized.equations: globalized.equations.pop(key, None)
            globalized = globalized | localized
        cls.__globalized__ = globalized
        cls.__localized__ = localized

    def __call__(cls, *args, **kwargs):
        functions = dict(cls.globalized.equations)
        outputs = tuple(cls.globalized.variables) if cls.globalized.variables else tuple(functions)
        parameters = {name for function in functions.values() for name in inspect.signature(function).parameters}
        defaults = cls.globalized.defaults | {name: kwargs[name] for name in parameters if name in kwargs}
        instance = super().__call__(*args, functions=functions, defaults=defaults, outputs=outputs, **kwargs)
        return instance

    @property
    def globalized(cls): return cls.__globalized__
    @property
    def localized(cls): return cls.__localized__


class Equation(Mixin, metaclass=EquationMeta):
    def __init__(self, *args, functions, defaults, outputs, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(functions, dict) and isinstance(defaults, dict)
        self.__functions = functions
        self.__defaults = defaults
        self.__outputs = outputs

    def execute(self, **parameters): return self.evaluate(**parameters)
    def evaluate(self, **parameters):
        domain = self.defaults | parameters
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
        outputs = {output: resolve(output) for output in self.outputs}
        return outputs

    @property
    def functions(self): return self.__functions
    @property
    def defaults(self): return self.__defaults
    @property
    def outputs(self): return self.__outputs


class DataframeEquation(Equation):
    def execute(self, dataframe, **parameters):
        if not isinstance(dataframe, pd.DataFrame): raise TypeError(type(dataframe))
        results = pd.DataFrame(index=dataframe.index)
        dependency = lambda parameter: parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        dependencies = {parameter.name for function in self.functions.values() for parameter in inspect.signature(function).parameters.values() if dependency(parameter)}
        dependencies = dependencies | (set(self.outputs) - self.functions.keys())
        series = {name: dataframe[name] for name in dependencies if name in dataframe.columns}
        parameters = series | parameters
        outputs = self.evaluate(**parameters)
        for name, series in outputs.items(): results[name] = series
        return results


