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
from itertools import chain
from functools import update_wrapper
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DelayerMeta", "SingletonMeta", "AttributeMeta", "RegistryMeta", "NamingMeta"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


aslist = lambda x: [x] if not isinstance(x, (list, tuple)) else list(x)
astuple = lambda x: (x,) if not isinstance(x, (list, tuple)) else tuple(x)
remove = lambda x, j: [i for i in x if i is not j]
flatten = lambda y: [i for x in y for i in x]
unique = lambda x: list(ODict.fromkeys(x))
insert = lambda x, i, j: x[:x.index(i)] + [j] + x[x.index(i):]
isnamed = lambda x: issubclass(x, tuple) and hasattr(x, "_fields") and all([isinstance(field, str) for field in getattr(x, "_fields")])
ismeta = lambda x, m: type(x) is m or issubclass(type(x), m)
isdunder = lambda x: str(x).startswith('__') and str(x).endswith('__')
isenum = lambda x: str(x).upper() == str(x) and not isdunder(x)
astype = lambda base, meta: meta(base.__name__, (base,), {})


class Meta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        try: return super(Meta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError: return super(Meta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        try: super(Meta, cls).__init__(name, bases, attrs, *args, **kwargs)
        except TypeError: super(Meta, cls).__init__(name, bases, attrs, *args, **kwargs)


class DelayerMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(DelayerMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        cls.__delay__ = kwargs.get("delay", getattr(cls, "__delay__", None))
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


class RegistryMeta(Meta):
    def __iter__(cls): return iter(list(cls.registry.items()))
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        assert "registry" not in attrs.keys()
        super(RegistryMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, RegistryMeta) for base in bases]):
            assert "register" not in kwargs.keys()
            cls.__registry__ = dict()
            return
        register = kwargs.get("register", [])
        register = [register] if not isinstance(register, list) else register
        register = list(filter(lambda value: value is not None, register))
        for key in register: cls[key] = cls

    def __setitem__(cls, key, value): cls.registry[key] = value
    def __getitem__(cls, key): return cls.registry[key]

    @property
    def registry(cls): return cls.__registry__


class AttributeMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        assert "root" not in attrs.keys()
        super(AttributeMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, AttributeMeta) for base in bases]):
            assert "attribute" not in kwargs.keys() and "attributes" not in kwargs.keys()
            cls.__root__ = cls
            return
        attributes = [kwargs.get("attribute", None)] + kwargs.get("attributes", [])
        attributes = list(filter(lambda attribute: attribute is not None, attributes))
        assert all([isinstance(attribute, str) for attribute in attributes])
        for attribute in attributes: setattr(cls.root, attribute, cls)

    @property
    def root(cls): return cls.__root__


class NamingMeta(Meta):
    def __init__(cls, *args, **kwargs):
        super(NamingMeta, cls).__init__(*args, **kwargs)
        cls.__fields__ = getattr(cls, "__fields__", []) + kwargs.get("fields", [])
        cls.__named__ = getattr(cls, "__named__", {}) | kwargs.get("named", {})

    def __iter__(cls): return chain(cls.fields, cls.named.keys())
    def __call__(cls, contents, *args, **kwargs):
        keys = chain(cls.fields, cls.named.keys())
        assert isinstance(contents, dict) and all([key in contents.keys() for key in keys])
        instance = super(NamingMeta, cls).__call__(*args, **kwargs)
        for attribute, value in cls.named.items():
            value = value(contents[attribute], *args, **kwargs)
            setattr(instance, attribute, value)
        for attribute in cls.fields:
            value = contents[attribute]
            setattr(instance, attribute, value)
        return instance

    @property
    def fields(cls): return cls.__fields__
    @property
    def named(cls): return cls.__named__

















