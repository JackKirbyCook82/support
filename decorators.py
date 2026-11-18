# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2017
@name    Function Decorators
@author: Jack Kirby Cook
@file:   support/decorators.py

"""

import copy
from abc import ABC, abstractmethod
from functools import update_wrapper

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Wrapper", "Decorator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Base(ABC):
    def __init__(self):
        self.__wrapped__ = None
        self.__self__ = None

    def __get__(self, instance, owner):
        if instance is None: return self
        bounded = copy.copy(self)
        bounded.instance = instance
        return bounded

    def invoke(self, instance, *args, **kwargs):
        function = self.function
        if isinstance(function, Base):
            function = function.__get__(instance, type(instance))
            return function(*args, **kwargs)
        return function(instance, *args, **kwargs)

    @property
    def function(self): return self.__wrapped__
    @property
    def instance(self): return self.__self__
    @instance.setter
    def instance(self, instance): self.__self__ = instance


class Wrapper(Base):
    def __init__(self, function):
        super().__init__()
        update_wrapper(self, function, updated=())

    def __call__(self, *args, **kwargs):
        if self.instance is None: return self.wrapper(*args, **kwargs)
        else: return self.wrapper(self.instance, *args, **kwargs)

    @abstractmethod
    def wrapper(self, *args, **kwargs): pass


class Decorator(Base):
    def __init__(self, *arguments, **parameters):
        super().__init__()
        self.__parameters = dict(parameters)
        self.__arguments = list(arguments)

    def __contains__(self, content):
        if isinstance(content, int): return -len(self.arguments) <= content < len(self.arguments)
        elif isinstance(content, str): return content in self.parameters.keys()
        else: return False

    def __getitem__(self, content):
        if isinstance(content, int): return self.arguments[content]
        elif isinstance(content, str): return self.parameters[content]
        else: raise TypeError(type(content))

    def __call__(self, *args, **kwargs):
        if self.function is None: return self.wrapper(*args, **kwargs)
        if self.instance is None: return self.decorator(*args, **kwargs)
        else: return self.decorator(self.instance, *args, **kwargs)

    def wrapper(self, function):
        assert callable(function)
        update_wrapper(self, function, updated=())
        return self

    @abstractmethod
    def decorator(self, *args, **kwargs): pass

    @property
    def parameters(self): return self.__parameters
    @property
    def arguments(self): return self.__arguments


