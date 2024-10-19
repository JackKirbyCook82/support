# -*- coding: utf-8 -*-
"""
Created on Thurs Cot 17 2024
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import inspect
from copy import copy
from abc import ABC, abstractmethod

from support.meta import SingletonMeta, RegistryMeta
from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Variable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Variable(Node, metaclass=RegistryMeta):
    def __new__(cls, *args, **kwargs):
        if issubclass(cls, Variable) and cls is not Variable:
            return object.__new__(cls)
        function = kwargs.get("function", None)
        return Variable[bool(function)](*args, **kwargs)

    def __init__(self, varname, vartype, *args, domain, **kwargs):
        super().__init__(*args, **kwargs)
        self.__domain = domain
        self.__type = vartype
        self.__name = varname
        self.__result = None
        self.__state = False

    def __bool__(self): return bool(self.state)
    def __str__(self): return str(self.name)
    def __len__(self): return int(self.size)

    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)

    def __call__(self, contents):
        if bool(self): return self.result
        self.result = self.execute(contents)
        self.state = True
        return self.result

    @abstractmethod
    def execute(self, contents): pass

    @property
    def domain(self): return self.__domain
    @property
    def result(self): return self.__results
    @property
    def state(self): return self.__state
    @property
    def name(self): return self.__name
    @property
    def type(self): return self.__type

    @result.setter
    def result(self, result): self.__result = result
    @state.setter
    def state(self, state): self.__state = state


class Equation(Variable, register=True):
    def __init__(self, *args, function, **kwargs):
        domain = list(inspect.signature(function).parameters.keys())
        super().__init__(*args, domain=domain, **kwargs)
        self.__function = function

    def execute(self, contents):
        domain = [child(contents) for child in self.children]
        return self.function(*domain)

    @property
    def function(self): return self.__function


class Source(Variable, register=False):
    def __init__(self, *args, locator, **kwargs):
        super().__(*args, domain=[], **kwargs)
        self.__locator = locator

    def execute(self, contents):
        return contents[self.locator]

    @property
    def locator(self): return self.__locator


class CalculationMeta(SingletonMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.items() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(CalculationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = {key: variable for key, variable in getattr(cls, "__variables__", {}).items()}
        updated = {key: variable for key, variable in attrs.items() if isinstance(variable, Variable)}
        cls.__variables__ = existing | updated

    def __call__(cls, *args, **kwargs):
        variables = {key: copy(variable) for key, variable in cls.__variables__.items()}
        for variable in variables.values():
            for key in list(variable.domain):
                variable[key] = variables[key]
        instance = super(CalculationMeta, cls).__call__(*args, variables=variables, **kwargs)
        return instance


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, *args, variables, **kwargs):
        self.__variables = variables

    def __call__(self, *args, **kwargs):
        pass

    @property
    def variables(self): return self.__variables



