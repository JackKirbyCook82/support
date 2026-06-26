# -*- coding: utf-8 -*-
"""
Created on Tues Mar 18 2025
@name:   Custom Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from numbers import Number
from dataclasses import dataclass
from collections import OrderedDict
from datetime import date as Date
from datetime import datetime as Datetime

from support.decorators import Dispatchers

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "NumRange", "SliceOrderedDict"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class DateRange:
    minimum: Date | Datetime; maximum: Date | Datetime

    @classmethod
    def create(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(value, (Date, Datetime)) for value in dates])
        if not dates: return None
        return cls(min(dates), max(dates))

    def __contains__(self, value): return self.minimum <= value <= self.maximum
    def __iter__(self): return iter(pd.date_range(start=self.minimum, end=self.maximum))
    def __str__(self): return f"{self.minimum}|{self.maximum}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


@dataclass(frozen=True)
class NumRange:
    minimum: float; maximum: float

    @classmethod
    def create(cls, numbers):
        assert isinstance(numbers, list)
        assert all([isinstance(number, Number) for number in numbers])
        if not numbers: return None
        return cls(min(numbers), max(numbers))

    def __contains__(self, value): return self.minimum <= value <= self.maximum
    def __str__(self): return f"{self.minimum}|{self.maximum}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return self.maximum - self.minimum


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








