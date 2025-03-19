# -*- coding: utf-8 -*-
"""
Created on Tues Mar 18 2025
@name:   Custom Objects
@author: Jack Kirby Cook

"""

from collections import OrderedDict

from support.decorators import TypeDispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SliceOrderedDict"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class SliceOrderedDict(OrderedDict):
    def __getitem__(self, key): return self.locate(key)

    @TypeDispatcher(locator=0)
    def pop(self, key, default=None): return super().pop(key, default)
    @pop.register(str)
    def popString(self, key, default=None): return super().pop(key, default)
    @pop.register(int)
    def popInteger(self, index, default=None):
        key = list(self.keys())[index]
        return super().pop(key, default)

    @TypeDispatcher(locator=0)
    def get(self, key, default=None): return super().get(key, default)
    @pop.register(str)
    def getString(self, key, default=None): return super().get(key, default)
    @pop.register(int)
    def getInteger(self, index, default=None):
        key = list(self.keys())[index]
        return super().get(key, default)

    @TypeDispatcher(locator=0)
    def locate(self, key): return super().__getitem__(key)
    @locate.register(str)
    def locateString(self, key): return super().__getitem__(key)

    @locate.register(int)
    def locateInteger(self, index):
        key = list(self.keys())[index]
        value = self.locate(key)
        return type(self)({key: value})

    @locate.register(slice)
    def locateSlice(self, indexes):
        items = list(self.items())[indexes]
        return type(self)(items)









