# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2017
@name    Function Dispatchers
@author: Jack Kirby Cook

"""

import copy
from abc import ABC, abstractmethod
from functools import update_wrapper

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Wrapper", "Decorator", "TypeDispatcher", "ValueDispatcher"]
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = "MIT License"


class Wrapper(object):
    def __init__(self, function):
        update_wrapper(self, function)
        self.__wrapped__ = function
        self.__self__ = None

    def __call__(self, *args, **kwargs):
        if self.instance is None: return self.wrapper(*args, **kwargs)
        else: return self.wrapper(self.instance, *args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None: return self
        bounded = copy.copy(self)
        bounded.instance = instance
        update_wrapper(bounded, self.function)
        attribute = self.function.__name__
        setattr(instance, attribute, bounded)
        return bounded

    def wrapper(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @property
    def function(self): return self.__wrapped__
    @property
    def instance(self): return self.__self__
    @instance.setter
    def instance(self, instance): self.__self__ = instance


class Decorator(object):
    def __init__(self, *arguments, **parameters):
        self.__parameters = dict(parameters)
        self.__arguments = list(arguments)
        self.__wrapped__ = None
        self.__self__ = None

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

    def __get__(self, instance, owner):
        if instance is None: return self
        bounded = copy.copy(self)
        bounded.instance = instance
        update_wrapper(self, self.function)
        attribute = self.function.__name__
        setattr(instance, attribute, bounded)
        return bounded

    def wrapper(self, function):
        assert callable(function)
        update_wrapper(self, function)
        return self

    def decorator(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @property
    def parameters(self): return self.__parameters
    @property
    def arguments(self): return self.__arguments
    @property
    def function(self): return self.__wrapped__
    @property
    def instance(self): return self.__self__
    @instance.setter
    def instance(self, instance): self.__self__ = instance


class Signature(Decorator):
    def __init__(self, signature, *args, **kwargs):
        assert isinstance(signature, str)
        assert "->" in str(signature)
        super().__init__(*args, **kwargs)
        inlet, outlet = str(signature).split("->")
        inlet = list(filter(bool, str(inlet).split(",")))
        inlet = [string for string in inlet if "*" not in string]
        optional = [str(string).strip("*") for string in inlet if "*" not in string]
        outlet = list(filter(bool, str(outlet).split(",")))
        self.__domain = list(inlet)
        self.__optional = list(optional)
        self.__range = list(outlet)

    @property
    def domain(self): return self.__domain
    @property
    def optional(self): return self.__optional
    @property
    def range(self): return self.__range


class Dispatcher(Decorator, ABC):
    def __init__(self, *args, locator, **kwargs):
        assert isinstance(locator, (int, str))
        super().__init__(*args, **kwargs)
        self.__locator = locator
        self.__registry = dict()

    def decorator(self, *args, **kwargs):
        method = bool(self.instance is not None)
        if isinstance(self.locator, int): locator = args[self.locator + int(method)]
        elif isinstance(self.locator, str): locator = kwargs.get(self.locator, None)
        else: raise TypeError(type(self.locator))
        locator = self.locate(locator)
        function = self.registry.get(locator, self.function)
        return function(*args, **kwargs)

    def register(self, *locators):
        def decorator(function):
            assert callable(function)
            registry = {locator: function for locator in locators}
            self.registry.update(registry)
            return function
        return decorator

    @staticmethod
    @abstractmethod
    def locate(locator): pass

    @property
    def registry(self): return self.__registry
    @property
    def locator(self): return self.__locator


class TypeDispatcher(Dispatcher):
    @staticmethod
    def locate(locator): return type(locator)


class ValueDispatcher(Dispatcher):
    @staticmethod
    def locate(locator): return locator



