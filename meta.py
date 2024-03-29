# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 2021
@name:   MetaTypes
@author: Jack Kirby Cook

"""

import time
import multiprocessing
from abc import ABCMeta
from inspect import isclass
from functools import update_wrapper
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VariantMeta", "DelayerMeta", "SingletonMeta", "RegistryMeta"]
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


class VariantKeyError(Exception):
    def __str__(self): return f"{self.__class__.__name__}[{self.args[0].__name__ if isclass(self.args[0]) else type(self.args[0]).__name__}]"

class VariantValueError(Exception):
    def __str__(self): return f"{self.__class__.__name__}[{mrostr(self.args[0])}]"

class VariantMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([ismeta(base, VariantMeta) for base in bases]):
            attrs = {**attrs, "__variant__": False, "registry": dict()}
            cls = super(VariantMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
            return cls
        variants = [kwargs.get("variant", None)] + kwargs.get("variants", [])
        variants = list(filter(None, variants))
        variant = any([base.__variant__ for base in bases if type(base) is VariantMeta]) or bool(variants)
        attrs = {**attrs, "__variant__": variant}
        cls = super(VariantMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        cls.registry.update({key: cls for key in variants})
        return cls

    def __call__(cls, *args, variant, **kwargs):
        assert variant is not None
        if variant not in cls.registry.keys():
            raise VariantKeyError(variant)
        base = cls.registry[variant]
        if not bool(cls.__variant__):
            cls = type(cls)(cls.__name__, (cls, base), {}, *args, **kwargs)
        try:
            instance = super(VariantMeta, cls).__call__(*args, **kwargs)
            return instance
        except TypeError:
            raise VariantValueError(cls)


class DelayerMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        assert not any([attr in attrs.keys() for attr in ("delay", "timer", "wait")])
        cls = super(DelayerMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, *args, **kwargs):
        cls.__delay__ = kwargs.get("delay", getattr(cls, "__delay__", None))
        cls.__mutex__ = multiprocessing.Lock()
        cls.__timer__ = None

    def wait(cls):
        with cls.__mutex__:
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
    def timer(cls): return cls.__timer__
    @timer.setter
    def timer(cls, timer): cls.__timer__ = timer


class SingletonMeta(ABCMeta):
    instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta.instances.keys():
            SingletonMeta.instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return SingletonMeta.instances[cls]


class RegistryMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([ismeta(base, RegistryMeta) for base in bases]):
            attrs = {**attrs, "registry": {}}
        cls = super(RegistryMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __setitem__(cls, key, value): cls.registry[key] = value
    def __getitem__(cls, key): return cls.registry[key]

    def __init__(cls, *args, key=None, keys=[], **kwargs):
        assert isinstance(keys, list)
        keys = keys + ([key] if key is not None else [])
        for key in keys:
            cls[key] = cls




