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
from itertools import product
from functools import update_wrapper
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VariantMeta", "DelayerMeta", "SingletonMeta", "AttributeMeta", "RegistryMeta", "FieldsMeta"]
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
        try:
            return super(Meta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            return super(Meta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, *args, **kwargs):
        try:
            super(Meta, cls).__init__(*args, **kwargs)
        except TypeError:
            super(Meta, cls).__init__()


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
        assert all([attr not in attrs.keys() for attr in ("variant", "registry")])
        super(VariantMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, VariantMeta) for base in bases]):
            cls.__registry__ = dict()
            cls.__variant__ = False
            return
        variants = [kwargs.get("variant", None)] + kwargs.get("variants", [])
        variants = list(filter(None, variants))
        registry = cls.registry | {key: cls for key in variants}
        variant = any([base.variant for base in bases if type(base) is VariantMeta]) or bool(variants)
        cls.__registry__ = registry
        cls.__variant__ = variant

    def __call__(cls, *args, variant, **kwargs):
        assert variant is not None
        if variant not in cls.registry.keys():
            raise VariantKeyError(variant)
        base = cls.registry[variant]
        if not bool(cls.variant):
            cls = type(cls)(cls.__name__, (cls, base), {}, *args, **kwargs)
        try:
            instance = super(VariantMeta, cls).__call__(*args, **kwargs)
            return instance
        except TypeError:
            raise VariantValueError(cls)

    @property
    def registry(cls): return cls.__registry__
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
            SingletonMeta.__instances__[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return SingletonMeta.__instances__[cls]


class AttributeMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, register=None, **kwargs):
        assert isinstance(register, (list, str, type(None)))
        super(AttributeMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, AttributeMeta) for base in bases]):
            return
        parents = [base for base in bases if ismeta(base, AttributeMeta)]
        register = list(filter(None, [register] if not isinstance(register, list) else register))
        assert all([isinstance(value, str) for value in register])
        for parent, key in product(parents, register):
            setattr(parent, key, cls)


class RegistryMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, register=None, **kwargs):
        assert "registry" not in attrs.keys()
        super(RegistryMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, RegistryMeta) for base in bases]):
            cls.__registry__ = dict()
            return
        register = list(filter(None, [register] if not isinstance(register, list) else register))
        registry = cls.registry | {key: cls for key in register}
        cls.__registry__ = registry

    def __setitem__(cls, key, value): cls.registry[key] = value
    def __getitem__(cls, key): return cls.registry[key]

    @property
    def registry(cls): return cls.__registry__


class FieldsMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, fields=[], **kwargs):
        assert all([attr not in attrs.keys() for attr in ("keys", "values", "items")])
        assert all([attr not in attrs.keys() for attr in ("todict", "tolist", "totuple")])
        assert "fields" not in attrs.keys() and isinstance(fields, list)
        super(FieldsMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([ismeta(base, FieldsMeta) for base in bases]):
            assert all([field not in attrs.keys() for field in fields])
            cls.__fields__ = fields
            return
        fields = [field for field in fields if field not in cls.fields]
        assert all([field not in attrs.keys() for field in fields])
        cls.__fields__ = fields

    def __call__(cls, *args, **kwargs):
        contents = ODict([(field, kwargs.get(field, None)) for field in cls.fields])
        instance = super(FieldsMeta, cls).__call__(*args, **kwargs)
        for key, value in contents.items():
            setattr(instance, key, value)
        setattr(instance, "totuple", lambda: tuple(contents.values()))
        setattr(instance, "tolist", lambda: list(contents.values()))
        setattr(instance, "todict", lambda: dict(contents))
        setattr(instance, "values", lambda: contents.values())
        setattr(instance, "items", lambda: contents.items())
        setattr(instance, "keys", lambda: contents.keys())
        return instance

    @property
    def fields(cls): return cls.__fields__



