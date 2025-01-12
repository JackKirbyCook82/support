# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Variables Object
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from datetime import date as Date
from datetime import datetime as Datetime
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Category", "Variables", "Variable"]
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


class VariableMeta(EnumMeta):
    def __iter__(cls): return iter([state for state in super().__iter__() if bool(state)])
    def __getitem__(cls, string):
        string = str(string).upper().replace(" ", "")
        return super(VariableMeta, cls).__getitem__(string)

    def __call__(cls, *args, **kwargs):
        if bool(cls._member_map_): return cls.retrieve(*args, **kwargs)
        else: return cls.create(*args, **kwargs)

    def create(cls, name, contents, *args, start=1, **kwargs):
        assert isinstance(start, int) and isinstance(contents, list) and all([isinstance(content, str) for content in contents])
        contents = list(map(lambda string: str(string).upper().replace(" ", ""), list(contents)))
        return super(VariableMeta, cls).__call__(name, contents, start=start)

    def retrieve(cls, content, *args, **kwargs):
        if isinstance(content, str) and str(content).isdigit():
            return super(VariableMeta, cls).__call__(int(content))
        elif isinstance(content, str) and not str(content).isdigit():
            content = str(content).upper()
            return super(VariableMeta, cls).__getitem__(str(content))
        return super(VariableMeta, cls).__call__(content)


class Variable(Enum, metaclass=VariableMeta):
    def __hash__(self): return hash((self.name, self.value))
    def __str__(self): return str(self.name).lower()
    def __bool__(self): return bool(self.value)
    def __int__(self): return int(self.value)


class Collection(ABC):
    def __init__(self, name, contents, parameters):
        self.__parameters = parameters
        self.__contents = contents
        self.__name = name

    def __int__(self): return int(sum([pow(10, index) * int(content) for index, content in enumerate(reversed(self.contents.values()))]))
    def __str__(self): return str("|".join([str(content) for content in iter(self.contents.values()) if bool(content)]))
    def __bool__(self): return any([bool(content) for content in iter(self.contents.values())])
    def __hash__(self): return hash(tuple(self.contents.items()))

    def __reversed__(self): return reversed(self.contents.values())
    def __iter__(self): return iter(self.contents.values())

    def items(self): return self.contents.items()
    def values(self): return self.contents.values()
    def keys(self): return self.contents.keys()

    @property
    def parameters(self): return self.__parameters
    @property
    def contents(self): return self.__contents
    @property
    def name(self): return self.__name


class Variables(ABCMeta):
    def __new__(mcs, dataname, datafields, dataparams=set()):
        cls = super(Variables, mcs).__new__(mcs, dataname, (Collection, ABC), {})
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
        instance = super(Variables, cls).__call__(name, contents, parameters)
        for attribute, value in contents.items():
            setattr(instance, attribute, value)
        for attribute, value in parameters.items():
            setattr(instance, attribute, value)
        return instance


class CategoryMeta(ABCMeta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(CategoryMeta, cls).__init__(name, bases, attrs)
        category = lambda value: issubclass(type(value), CategoryMeta) or type(value) is CategoryMeta
        collection = lambda value: isinstance(value, Collection)
        variable = lambda value: isinstance(value, Variable)
        categories = {key: value for key, value in attrs.items() if category(value)}
        collections = {key: value for key, value in attrs.items() if collection(value)}
        variables = {key: value for key, value in attrs.items() if variable(value)}
        cls.__categories__ = getattr(cls, "__categories__", {}) | dict(categories)
        cls.__collections__ = getattr(cls, "__collections__", {}) | dict(collections)
        cls.__variables__ = getattr(cls, "__variables__", {}) | dict(variables)

    def __iter__(cls):
        for variable in cls.variables.values(): yield variable
        for collection in cls.collections.values(): yield collection
        for category in cls.categories.values(): yield from iter(category)

    def __getitem__(cls, string): return cls.strings[string]
    def __call__(cls, content):
        content = int(content) if str(content).isdigit() else content
        if isinstance(content, Collection): return cls.encodings[hash(content)]
        elif isinstance(content, Variable): return cls.encodings[hash(content)]
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
    def categories(cls): return cls.__categories__
    @property
    def collections(cls): return cls.__collections__
    @property
    def variables(cls): return cls.__variables__


class Category(ABC, metaclass=CategoryMeta):
    pass


