# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Concept Object
@author: Jack Kirby Cook

"""

import numbers
import pandas as pd
from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from datetime import date as Date
from datetime import datetime as Datetime
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "NumRange", "Concept", "Concepts", "Assembly"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, (Date, Datetime)) for date in dates])
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __iter__(self): return (date for date in pd.date_range(start=self.minimum, end=self.maximum))
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


class NumRange(ntuple("NumRange", "minimum maximum")):
    def __contains__(self, num): return self.minimum <= num <= self.maximum
    def __new__(cls, nums):
        assert isinstance(nums, list)
        assert all([isinstance(num, numbers.Number) for num in nums])
        return super().__new__(cls, min(nums), max(nums)) if nums else None

    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return self.maximum - self.minimum


class Collection(ABC):
    def __init__(self, name, contents, parameters):
        self.__parameters = parameters
        self.__contents = contents
        self.__name = name

    def __int__(self): return int(sum([pow(10, index) * int(content) for index, content in enumerate(reversed(self.contents.values()))]))
    def __str__(self): return str("|".join([str(content) for content in iter(self.contents.values()) if bool(content)]))
    def __bool__(self): return any([bool(content) for content in iter(self.contents.values())])
    def __hash__(self): return hash(tuple(self.contents.items()))
    def __iter__(self): return iter(list(self.contents.items()))

    def items(self): return self.contents.items()
    def values(self): return self.contents.values()
    def keys(self): return self.contents.keys()

    @property
    def parameters(self): return self.__parameters
    @property
    def contents(self): return self.__contents
    @property
    def name(self): return self.__name


class ConceptMeta(EnumMeta):
    def __iter__(cls): return iter([state for state in super().__iter__() if bool(state)])
    def __getitem__(cls, string):
        string = str(string).upper().replace(" ", "")
        return super(ConceptMeta, cls).__getitem__(string)

    def __call__(cls, *args, **kwargs):
        if bool(cls._member_map_): return cls.retrieve(*args, **kwargs)
        else: return cls.create(*args, **kwargs)

    def create(cls, name, contents, *args, start=1, **kwargs):
        assert isinstance(start, int) and isinstance(contents, list) and all([isinstance(content, str) for content in contents])
        contents = list(map(lambda string: str(string).upper().replace(" ", ""), list(contents)))
        return super(ConceptMeta, cls).__call__(name, contents, start=start)

    def retrieve(cls, content, *args, **kwargs):
        if isinstance(content, str) and str(content).isdigit():
            return super(ConceptMeta, cls).__call__(int(content))
        elif isinstance(content, str) and not str(content).isdigit():
            content = str(content).upper()
            return super(ConceptMeta, cls).__getitem__(str(content))
        return super(ConceptMeta, cls).__call__(content)


class Concept(Enum, metaclass=ConceptMeta):
    def __hash__(self): return hash((self.name, self.value))
    def __str__(self): return str(self.name).lower()
    def __bool__(self): return bool(self.value)
    def __int__(self): return int(self.value)


class Concepts(ABCMeta):
    def __new__(mcs, dataname, datafields, dataparams=set()):
        cls = super(Concepts, mcs).__new__(mcs, dataname, (Collection, ABC), {})
        return cls

    def __reversed__(cls): return reversed(cls.datafields)
    def __iter__(cls): return iter(cls.datafields)
    def __hash__(cls): return hash(tuple(cls.datafields))
    def __len__(cls): return len(cls.datafields)

    def __init__(cls, dataname, datafields, dataparams=set()):
        assert bool(dataname == cls.__name__) and isinstance(datafields, list) and isinstance(dataparams, set)
        assert all([isinstance(datafield, str) for datafield in datafields])
        assert all([isinstance(dataparam, str) for dataparam in dataparams])
        cls.dataparams = dataparams
        cls.datafields = datafields
        cls.dataname = dataname

    def __call__(cls, name, contents, *args, **kwargs):
        assert isinstance(contents, list)
        assert len(contents) == len(cls.datafields)
        contents = ODict([(field, content) for field, content in zip(cls.datafields, contents)])
        parameters = ODict([(parameter, kwargs[parameter]) for parameter in cls.dataparams])
        instance = super(Concepts, cls).__call__(name, contents, parameters)
        for attribute, value in contents.items():
            setattr(instance, attribute, value)
        for attribute, value in parameters.items():
            setattr(instance, attribute, value)
        return instance


class AssemblyMeta(ABCMeta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(AssemblyMeta, cls).__init__(name, bases, attrs)
        assembly = lambda value: issubclass(type(value), AssemblyMeta) or type(value) is AssemblyMeta
        collection = lambda value: isinstance(value, Collection)
        concept = lambda value: isinstance(value, Concept)
        assemblies = {key: value for key, value in attrs.items() if assembly(value)}
        collections = {key: value for key, value in attrs.items() if collection(value)}
        concepts = {key: value for key, value in attrs.items() if concept(value)}
        cls.__assemblies__ = getattr(cls, "__assemblies__", {}) | dict(assemblies)
        cls.__collections__ = getattr(cls, "__collections__", {}) | dict(collections)
        cls.__concepts__ = getattr(cls, "__concepts__", {}) | dict(concepts)

    def __iter__(cls):
        for concept in cls.concepts.values(): yield concept
        for collection in cls.collections.values(): yield collection
        for assembly in cls.assemblies.values(): yield from iter(assembly)

    def __getitem__(cls, string): return cls.strings[string]
    def __call__(cls, content):
        content = int(content) if str(content).isdigit() else content
        if isinstance(content, Collection): return cls.encodings[hash(content)]
        elif isinstance(content, Concept): return cls.encodings[hash(content)]
        elif isinstance(content, list): return cls.values[tuple(content)]
        elif isinstance(content, tuple): return cls.values[content]
        elif isinstance(content, int): return cls.numbers[content]
        elif isinstance(content, str): return cls.strings[content]
        else: raise TypeError(type(content))

    @property
    def values(cls): return {tuple(content.values()): content for content in iter(cls)}
    @property
    def numbers(cls): return {int(content): content for content in iter(cls)}
    @property
    def strings(cls): return {str(content): content for content in iter(cls)}
    @property
    def encodings(cls): return {hash(content): content for content in iter(cls)}

    @property
    def assemblies(cls): return cls.__assemblies__
    @property
    def collections(cls): return cls.__collections__
    @property
    def concepts(cls): return cls.__concepts__


class Assembly(ABC, metaclass=AssemblyMeta):
    pass


