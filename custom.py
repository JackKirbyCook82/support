# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2017
@name    Custom Objects
@author: Jack Kirby Cook

"""

from abc import ABC, ABCMeta
from collections import OrderedDict as ODict

from utilities.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["NamedCollection", "SliceOrderedDict"]
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


class NamedCollectionMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        try:
            cls = super(NamedCollectionMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            cls = super(NamedCollectionMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, **kwargs):
        fields = [field for field in getattr(cls, "__fields__", [])]
        update = kwargs.get("fields", [])
        assert not any([field in fields for field in update])
        assert isinstance(fields, list) and isinstance(update, list)
        cls.__fields__ = fields + update

    def __call__(cls, *args, **kwargs):
        fields = [field for field in cls.__fields__]
        if bool(args):
            assert len(args) == len(fields)
            contents = [arg for arg in args]
        elif bool(kwargs):
            contents = [kwargs.get(field, None) for field in fields]
        else:
            contents = [None] * len(fields)
        instance = super(NamedCollectionMeta, cls).__call__(fields, contents)
        return instance


class NamedCollection(ABC, metaclass=NamedCollectionMeta):
    def __init__(self, fields, contents):
        assert isinstance(fields, list) and isinstance(contents, list)
        assert len(fields) == len(contents)
        self.__fields = fields
        self.__contents = contents

    def __str__(self):
        contents = {key: self.formatter(value) for key, value in self.items() if value is not None}
        contents = {key: str(value) for key, value in contents.items()}
        strings = ["=".join([key, value]) for key, value in contents.items()]
        return ", ".join(strings)

    def __contains__(self, key): return key in self.keys() and getattr(self, key) is not None
    def __eq__(self, other): return self.items() == other.items()
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(self.items())

    def __getattr__(self, attr):
        if attr not in self.fields:
            raise AttributeError(attr)
        index = list(self.keys()).index(attr)
        value = list(self.values())[index]
        return value

    def __getitem__(self, key):
        index = list(self.keys()).index(key) if isinstance(key, str) else int(key)
        return list(self.contents)[index]

    def __setitem__(self, key, value):
        index = list(self.keys()).index(key) if isinstance(key, str) else int(key)
        self.contents[index] = value

    def items(self): return tuple([(key, value) for key, value in zip(self.keys(), self.values())])
    def values(self): return tuple(self.contents)
    def keys(self): return tuple(self.fields)

    @staticmethod
    def formatter(value): return value
    @property
    def fields(self): return self.__fields
    @property
    def contents(self): return self.__contents


class SliceOrderedDict(ODict):
    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            return self.read(key)
        elif isinstance(key, int):
            return self.retrieve(key, False)
        else:
            raise TypeError(type(key).__name__)

    @typedispatcher
    def read(self, key): raise TypeError(type(key).__name__)

    @read.register(slice)
    def readSlice(self, key):
        start, stop = key.start, key.stop 
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if stop < 0:
            stop = len(self) + stop
        instance = self.__class__()
        for index, key in enumerate(self.keys()): 
            if start <= index < stop: 
                instance[key] = self[key]
        return instance

    @read.register(int)
    def readInt(self, index):
        if index >= 0:
            return self.readslice(slice(index, index + 1))
        else:
            return self.readslice(slice(len(self) + index, len(self) + index + 1))

    def retrieve(self, key, pop):
        if abs(key + 1 if key < 0 else key) >= len(self):
            raise IndexError(key)
        key, value = list(*self.read(key).items())
        if pop:
            del self[key]
        return value

    def pop(self, key, default=None):
        if isinstance(key, str):
            return super().pop(key, default)
        elif isinstance(key, int):
            return self._retrieve(key, pop=True)
        else:
            raise TypeError(type(key).__name__)

    def get(self, key, default=None):
        if isinstance(key, str):
            return super().get(key, default)
        elif isinstance(key, int):
            return self._retrieve(key, pop=False)
        else:
            raise TypeError(type(key).__name__)
        
    def update(self, others, inplace=True):
        if not isinstance(others, dict):
            raise TypeError(type(others).__name__)
        updated = [(key, others.pop(key, value)) for key, value in self.items()]
        added = [(key, value) for key, value in others.items()]
        if not inplace:
            return self.__class__(updated + added)
        self = self.__class__(updated + added)
        return self






