# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2017
@name    Function Dispatchers
@author: Jack Kirby Cook

"""

from inspect import isclass
from functools import update_wrapper
from abc import ABC, abstractmethod
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["argsdispatcher", "kwargsdispatcher", "typedispatcher", "valuedispatcher"]
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = "MIT License"


class BaseRegistry(ODict, ABC):
    def __contains__(self, key):
        key = self.execute(key)
        return True if key is not None else False

    def __getitem__(self, key):
        key = self.execute(key)
        return super().__getitem__(key)

    @abstractmethod
    def execute(self, key): pass


class KeyRegistry(BaseRegistry):
    def execute(self, key): return hash(key) if hash(key) in self.keys() else None


class ValueRegistry(BaseRegistry):
    def execute(self, key):
        try:
            return key if key in self.keys() else None
        except TypeError:
            return None


class TypeRegistry(BaseRegistry):
    def execute(self, key): return type(key) if type(key) in self.keys() else None


class MRORegistry(BaseRegistry):
    def mro(self, key): return [base for base in (key.__mro__ if isclass(key) else type(key).__mro__) if base in self.keys()]
    def execute(self, key): return self.mro(key)[0] if bool(self.mro(key)) else None


class Registry(object):
    def __init__(self, function):
        registry = [("value", ValueRegistry()), ("type", TypeRegistry()), ("mro", MRORegistry())]
        self.__registry = ODict(registry)
        self.__function = function

    def __contains__(self, key):
        return any([key in registry for registry in self.registry.values()])

    def __getitem__(self, key):
        for registry in self.registry.values():
            if key in registry:
                return registry[key]
        return self.function

    def key(self, *keys): return self.decorator("key", *[hash(key) for key in keys])
    def value(self, *keys): return self.decorator("value", *[key for key in keys])
    def type(self, *keys): return self.decorator("type", *[key for key in keys])
    def mro(self, *keys): return self.decorator("mro", *[key for key in keys])

    def decorator(self, registry, *keys):
        assert registry in self.registry.keys()
        assert all([hasattr(key, "__hash__") for key in keys])

        def wrapper(function):
            assert callable(function)
            update = {key: function for key in keys}
            self.registry[registry].update(update)
            return function
        return wrapper

    @property
    def registry(self): return self.__registry
    @property
    def function(self): return self.__function


def argsdispatcher(index=0, parser=lambda x: x):
    assert isinstance(index, int)

    def decorator(mainfunction):
        assert callable(mainfunction)
        __method__ = True if "." in str(mainfunction.__qualname__) else False
        __registry__ = Registry(mainfunction)

        def method_wrapper(self, *args, **kwargs):
            lookup = parser(args[index])
            function = __registry__[lookup]
            return function(self, *args, **kwargs)

        def function_wrapper(*args, **kwargs):
            lookup = parser(args[index])
            function = __registry__[lookup]
            return function(*args, **kwargs)

        wrapper = method_wrapper if __method__ else function_wrapper
        wrapper.register = __registry__
        update_wrapper(wrapper, mainfunction)
        return wrapper
    return decorator


def kwargsdispatcher(key, parser=lambda x: x):
    assert isinstance(key, str)

    def decorator(mainfunction):
        assert callable(mainfunction)
        __method__ = True if "." in str(mainfunction.__qualname__) else False
        __registry__ = Registry(mainfunction)

        def method_wrapper(self, *args, **kwargs):
            lookup = parser(kwargs[key])
            function = __registry__[lookup]
            return function(self, *args, **kwargs)

        def function_wrapper(*args, **kwargs):
            lookup = parser(kwargs[key])
            function = __registry__[lookup]
            return function(*args, **kwargs)

        wrapper = method_wrapper if __method__ else function_wrapper
        wrapper.register = __registry__
        update_wrapper(wrapper, mainfunction)
        return wrapper
    return decorator


def typedispatcher(mainfunction):
    assert callable(mainfunction)
    __method__ = True if "." in str(mainfunction.__qualname__) else False
    __registry__ = {}

    def retrieve(key): return __registry__.get(key, mainfunction)
    def update(items): __registry__.update(items)

    def register(*keys):
        def decorate(function):
            assert callable(function)
            update({key: function for key in keys})
            return function
        return decorate

    def method_wrapper(self, *args, **kwargs):
        try:
            function = retrieve(type(args[0]))
            return function(self, args[0], *args[1:], **kwargs)
        except IndexError:
            return mainfunction(self, *args, **kwargs)

    def function_wrapper(*args, **kwargs):
        try:
            function = retrieve(type(args[0]))
            return function(args[0], *args[1:], **kwargs)
        except IndexError:
            return mainfunction(*args, **kwargs)

    wrapper = method_wrapper if __method__ else function_wrapper
    wrapper.retrieve = retrieve
    wrapper.update = update
    wrapper.register = register
    update_wrapper(wrapper, mainfunction)
    return wrapper


def valuedispatcher(mainfunction):
    assert callable(mainfunction)
    __method__ = True if "." in str(mainfunction.__qualname__) else False
    __registry__ = {}

    def retrieve(key): return __registry__.get(key, mainfunction)
    def update(items): __registry__.update(items)

    def register(*keys):
        def decorate(function):
            assert callable(function)
            update({key: function for key in keys})
            return function
        return decorate

    def method_wrapper(self, *args, **kwargs):
        try:
            function = retrieve(args[0])
            return function(self, *args[1:], **kwargs)
        except IndexError:
            return mainfunction(self, *args, **kwargs)

    def function_wrapper(*args, **kwargs):
        try:
            function = retrieve(args[0])
            return function(*args[1:], **kwargs)
        except IndexError:
            return mainfunction(*args, **kwargs)

    wrapper = method_wrapper if __method__ else function_wrapper
    wrapper.retrieve = retrieve
    wrapper.update = update
    wrapper.register = register
    update_wrapper(wrapper, mainfunction)
    return wrapper



