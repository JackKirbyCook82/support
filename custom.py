# -*- coding: utf-8 -*-
"""
Created on Tues Mar 18 2025
@name:   Custom Objects
@author: Jack Kirby Cook

"""

from inspect import isclass
from typing import Iterable
from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Mapping
from datetime import date as Date
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta

from support.decorators import Dispatchers

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SliceOrderedDict", "ReversibleDict", "NumberRange", "DateRange"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ValueTypeError(Exception): pass
class ValueOrderError(Exception): pass
class ValueMeta(type):
    def __new__(mcs, name, bases, attrs, *args, valuetype=tuple(), **kwargs):
        assert isinstance(valuetype, tuple) or isclass(valuetype)
        cls = super(ValueMeta, mcs).__new__(mcs, name, bases, attrs, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, valuetype=tuple(), **kwargs):
        super(ValueMeta, cls).__init__(name, bases, attrs, **kwargs)
        valuetype = valuetype if isinstance(valuetype, tuple) else tuple([valuetype])
        valuetype = getattr(cls, "__valuetype__", tuple()) + valuetype
        cls.__valuetype__ = valuetype

    def __call__(cls, *arguments, **parameters):
        assert bool(cls.valuetype)
        if len(arguments) == 1 and not parameters and isinstance(arguments[0], Iterable):
            values = tuple(arguments[0])
            if not values: return None
            arguments = (min(values), max(values))
        instance = super(ValueMeta, cls).__call__(*arguments, **parameters)
        if not isinstance(instance.minimum, cls.valuetype): raise ValueTypeError(type(instance.minimum))
        if not isinstance(instance.maximum, cls.valuetype): raise ValueTypeError(type(instance.maximum))
        if instance.minimum > instance.maximum: raise ValueTypeError()
        return instance

    @property
    def valuetype(cls): return cls.__valuetype__


@dataclass(frozen=True)
class ValueRange(metaclass=ValueMeta):
    minimum: Date | Datetime | Timedelta | float
    maximum: Date | Datetime | Timedelta | float

    def __iter__(self): return iter((self.minimum, self.maximum))
    def __str__(self): return f"{self.minimum}|{self.maximum}"
    def __bool__(self): return self.minimum < self.maximum

    def __contains__(self, value): return self.minimum <= value <= self.maximum
    def __add__(self, other):
        if other is None: return self
        cls = type(self)
        assert isinstance(other, cls)
        minimum = min(self.minimum, other.minimum)
        maximum = max(self.maximum, other.maximum)
        return cls(minimum=minimum, maximum=maximum)


class NumberRange(ValueRange, valuetype=[int, float]): pass
class DateRange(ValueRange, valuetype=(Date, Datetime)): pass


class ReversibleDict(Mapping):
    def __len__(self): return len(self.forward)
    def __init__(self, forward):
        assert isinstance(forward, dict)
        assert len(forward.values()) == len(set(forward.values()))
        self.__backward = {value: key for key, value in forward.items()}
        self.__forward = forward

    def __iter__(self): return iter(self.forward)
    def __reversed__(self): return iter(self.backward)

    def __getitem__(self, couple):
        key, reverse = couple
        assert isinstance(reverse, bool)
        return self.get(key, reverse, default=None)

    def get(self, key, reverse=None, default=None):
        if reverse: return self.backward.get(key, default)
        else: return self.forward.get(key, default)

    @property
    def forward(self): return self.__forward
    @property
    def backward(self): return self.__backward


class SliceOrderedDict(OrderedDict):
    def __getitem__(self, key): return self.locate(key)

    @Dispatchers.Type(locator=0)
    def pop(self, key, default=None): return super().pop(key, default)

    @pop.register(str)
    def _(self, key, default=None): return super().pop(key, default)

    @pop.register(int)
    def _(self, index, default=None):
        key = list(self.keys())[index]
        return super().pop(key, default)

    @Dispatchers.Type(locator=0)
    def get(self, key, default=None): return super().get(key, default)

    @pop.register(str)
    def _(self, key, default=None): return super().get(key, default)

    @pop.register(int)
    def _(self, index, default=None):
        key = list(self.keys())[index]
        return super().get(key, default)

    @Dispatchers.Type(locator=0)
    def locate(self, key): return super().__getitem__(key)

    @locate.register(str)
    def _(self, key): return super().__getitem__(key)

    @locate.register(int)
    def _(self, index):
        key = list(self.keys())[index]
        value = self.locate(key)
        return type(self)({key: value})

    @locate.register(slice)
    def _(self, indexes):
        items = list(self.items())[indexes]
        return type(self)(items)








