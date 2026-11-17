# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2017
@name    Function Dispatchers
@author: Jack Kirby Cook
@file:   support/dispatchers.py

"""

from abc import ABC, abstractmethod

from support.decorators import Decorator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TypeDispatcher", "ValueDispatcher"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


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




