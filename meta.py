# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 2021
@name:   MetaTypes
@author: Jack Kirby Cook

"""

import time
import types
import multiprocessing
from abc import ABCMeta
from inspect import isclass
from functools import update_wrapper
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VariantMeta", "DelayerMeta", "SingletonMeta", "ParametersMeta", "AttributeMeta", "RegistryMeta"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


aslist = lambda x: [x] if not isinstance(x, (list, tuple)) else list(x)
astuple = lambda x: (x,) if not isinstance(x, (list, tuple)) else tuple(x)
remove = lambda x, j: [i for i in x if i is not j]
flatten = lambda y: [i for x in y for i in x]
unique = lambda x: list(ODict.fromkeys(x))
insert = lambda x, i, j: x[:x.index(i)] + [j] + x[x.index(i):]
ismeta = lambda x, m: type(x) is m or issubclass(type(x), m)
isdunder = lambda x: str(x).startswith('__') and str(x).endswith('__')
isenum = lambda x: str(x).upper() == str(x) and not isdunder(x)
mrostr = lambda x: ", ".join(list(map(lambda i: i.__name__, x.__mro__)))
astype = lambda base, meta: meta(base.__name__, (base,), {})


class Meta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        return super(Meta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(Meta, cls).__init__(name, bases, attrs, *args, **kwargs)


class VariantKeyError(Exception):
    def __str__(self): return f"{self.name}[{self.key}]"
    def __init__(self, key):
        self.__key = key.__name__ if isclass(key) else type(key).__name__
        self.__name = self.__class__.__name__

    @property
    def name(self): return self.__name
    @property
    def key(self): return self.__key


class VariantValueError(Exception):
    def __str__(self): return f"{self.name}[{self.value}]"
    def __init__(self, value):
        self.__value = mrostr(value)
        self.__name = self.__class__.__name__

    @property
    def name(self): return self.__name
    @property
    def value(self): return self.__value


class VariantMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        assert all([attr not in attrs.keys() for attr in ("variant", "variants")])
        super(VariantMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, VariantMeta) for base in bases]):
            cls.__variants__ = dict()
            cls.__variant__ = False
            return
        variants = [kwargs.get("variant", None)] + kwargs.get("variants", [])
        variants = list(filter(None, variants))
        variant = any([base.variant for base in bases if type(base) is VariantMeta]) or bool(variants)
        variants = cls.variants | {key: cls for key in variants}
        cls.__variants__ = variants
        cls.__variant__ = variant

    def __call__(cls, *args, variant, **kwargs):
        assert variant is not None
        if variant not in cls.variants.keys():
            raise VariantKeyError(variant)
        base = cls.variants[variant]
        if not bool(cls.variant):
            cls = type(cls)(cls.__name__, (cls, base), {}, *args, **kwargs)
        try:
            instance = super(VariantMeta, cls).__call__(*args, **kwargs)
            return instance
        except TypeError:
            raise VariantValueError(cls)

    @property
    def variants(cls): return cls.__variants__
    @property
    def variant(cls): return cls.__variant__


class DelayerMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(DelayerMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        cls.__delay__ = kwargs.get("delay", getattr(cls, "delay", None))
        cls.__mutex__ = multiprocessing.Lock()
        cls.__timer__ = None

    def wait(cls):
        with cls.mutex:
            if bool(cls.timer) and bool(cls.delay):
                seconds = (Datetime.now() - cls.timer).total_seconds()
                wait = max(cls.delay - seconds, 0)
                time.sleep(wait)
            cls.timer = Datetime.now()

    @staticmethod
    def delayer(function):
        assert "." in function.__qualname__

        def wrapper(self, *args, **kwargs):
            type(self).wait()
            return function(self, *args, **kwargs)

        update_wrapper(wrapper, function)
        return wrapper

    @property
    def delay(cls): return cls.__delay__
    @property
    def mutex(cls): return cls.__mutex__
    @property
    def timer(cls): return cls.__timer__
    @timer.setter
    def timer(cls, timer): cls.__timer__ = timer


class SingletonMeta(Meta):
    __instances__ = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta.__instances__.keys():
            instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
            SingletonMeta.__instances__[cls] = instance
        return SingletonMeta.__instances__[cls]


class ParametersMeta(Meta):
    def __iter__(cls): return iter(list(cls.parameters.items()))
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        function = lambda value: isinstance(value, types.LambdaType) or not isinstance(value, (types.MethodType, types.FunctionType))
        super(ParametersMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = getattr(cls, "__parameters__", {})
        update = {key: value for key, value in attrs.items() if function(value)}
        cls.__parameters__ = existing | update

    @property
    def parameters(cls): return cls.__parameters__


class RegistryMeta(Meta):
    def __iter__(cls): return iter(list(cls.registry.items()))
    def __init__(cls, name, bases, attrs, *args, register=None, **kwargs):
        assert "registry" not in attrs.keys()
        super(RegistryMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, RegistryMeta) for base in bases]):
            assert register is None
            cls.__registry__ = dict()
            return
        register = list(filter(None, [register] if not isinstance(register, list) else register))
        registry = cls.registry | {key: cls for key in register}
        for register in registry:
            cls[register] = cls

    def __setitem__(cls, key, value): cls.registry[key] = value
    def __getitem__(cls, key): return cls.registry[key]

    @property
    def registry(cls): return cls.__registry__


class AttributeMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, attribute=None, **kwargs):
        assert "root" not in attrs.keys()
        assert isinstance(attribute, (list, str, type(None)))
        super(AttributeMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, AttributeMeta) for base in bases]):
            assert attribute is None
            cls.__root__ = cls
            return
        attributes = list(filter(None, [attribute] if not isinstance(attribute, list) else attribute))
        assert all([isinstance(attribute, str) for attribute in attributes])
        for attribute in attributes:
            setattr(cls.root, attribute, cls)

    @property
    def root(cls): return cls.__root__



