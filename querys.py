# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Query Object
@author: Jack Kirby Cook

"""

import logging
from numbers import Number
from functools import total_ordering
from datetime import date as Date
from datetime import datetime as Datetime
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Query", "Field"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class FieldData(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.dataparams = getattr(cls, "dataparams", {}) | kwargs.get("dataparams", {})
        cls.datatype = kwargs.get("datatype", getattr(cls, "datatype", None))

    @staticmethod
    @abstractmethod
    def parse(string, *args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def encode(value, *args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def string(value, *args, **kwargs): pass


class FieldString(FieldData, datatype=str, dataparams={}):
    @staticmethod
    def parse(string, *args, **kwargs): return str(string)
    @staticmethod
    def encode(value, *args, **kwargs): return hash(value)
    @staticmethod
    def string(value, *args, **kwargs): return str(value)

class FieldNumber(FieldData, datatype=Number, dataparams={"digits": 2}):
    @staticmethod
    def parse(string, *args, digits, **kwargs): return round(string, digits)
    @staticmethod
    def encode(value, *args, **kwargs): return hash(value)
    @staticmethod
    def string(value, *args, digits, **kwargs): return str(round(value, digits))

class FieldDate(FieldData, datatype=Date, dataparams={"formatting": "%Y-%m-%d"}):
    @staticmethod
    def parse(string, *args, formatting, **kwargs): return Datetime.strptime(string, formatting)
    @staticmethod
    def encode(value, *args, **kwargs): return hash(Datetime(year=value.year, month=value.month, day=value.day).timestamp())
    @staticmethod
    def string(value, *args, formatting, **kwargs): return str(value.strftime(formatting))


class FieldMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        mixins = {subclass.datatype: subclass for subclass in FieldData.__subclasses__()}
        datatype = kwargs.get("datatype", None)
        if datatype is not None: bases = tuple([mixins[datatype]] + list(bases))
        cls = super(FieldMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __str__(cls): return str(cls.__dataname__)
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is FieldMeta for base in cls.__bases__]):
            return
        dataparams = {key: kwargs.get(key, default) for key, default in cls.dataparams.items()}
        cls.__dataparams__ = getattr(cls, "__dataparams__", {}) | dataparams
        cls.__dataname__ = kwargs.get("dataname", getattr(cls, "__dataname__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))

    def __getitem__(cls, datastring):
        dataname, datatype, dataparams = cls.__dataname__, cls.__datatype__, cls.__dataparams__
        assert isinstance(datastring, str)
        datavalue = cls.parse(datastring, **dataparams)
        assert isinstance(datavalue, datatype)
        instance = super(FieldMeta, cls).__call__(dataname, datavalue, dataparams)
        return instance

    def __call__(cls, datavalue):
        dataname, datatype, dataparams = cls.__dataname__, cls.__datatype__, cls.__dataparams__
        assert isinstance(datavalue, datatype)
        instance = super(FieldMeta, cls).__call__(dataname, datavalue, dataparams)
        return instance

    @staticmethod
    @abstractmethod
    def parse(string, *args, **kwargs): pass


@total_ordering
class Field(ABC, metaclass=FieldMeta):
    def __init__(self, name, value, parameters):
        self.__parameters = parameters
        self.__value = value
        self.__name = name

    def __hash__(self): return self.encode(self.value, **self.parameters)
    def __str__(self): return self.string(self.value, **self.parameters)

    def __eq__(self, other):
        assert type(other) == type(self)
        return self.value == other.value

    def __lt__(self, other):
        assert type(other) == type(self)
        return self.value < other.value

    @staticmethod
    @abstractmethod
    def encode(value, *args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def string(value, *args, **kwargs): pass

    @property
    def parameters(self): return self.__parameters
    @property
    def value(self): return self.__value
    @property
    def name(self): return self.__name


class QueryMeta(ABCMeta):
    def __iter__(cls): return map(str, cls.__fields__)
    def __init__(cls, *args, **kwargs):
        cls.__delimiter__ = kwargs.get("delimiter", getattr(cls, "__delimiter__", "|"))
        cls.__fields__ = getattr(cls, "__fields__", []) + kwargs.get("fields", [])

    def __getitem__(cls, strings):
        assert isinstance(strings, str)
        fields, delimiter = cls.__fields__, cls.__delimiter__
        mapping = cls.create(strings, fields=fields, delimiter=delimiter)
        contents = list(mapping.values())
        instance = super(QueryMeta, cls).__call__(contents, delimiter=delimiter)
        for attribute, content in mapping.items():
            setattr(instance, attribute, content)
        return instance

    def __call__(cls, values):
        if isinstance(values, Query):
            values = ODict(values.items())
        fields, delimiter = cls.__fields__, cls.__delimiter__
        mapping = cls.create(values, fields=fields, delimiter=delimiter)
        contents = list(mapping.values())
        instance = super(QueryMeta, cls).__call__(contents, delimiter=delimiter)
        for attribute, content in mapping.items():
            setattr(instance, attribute, content)
        return instance

    @typedispatcher
    def create(cls, values): raise TypeError(type(values))

    @create.register(str)
    def create_string(cls, values, *args, fields, delimiter, **kwargs):
        strings = str(values).split(delimiter)
        assert len(strings) == len(cls)
        contents = [field[string] for field, string in zip(fields, strings)]
        return ODict([(field, content) for field, content in zip(list(cls), contents)])

    @create.register(list)
    def create_collection(cls, values, *args, fields, **kwargs):
        assert len(values) == len(cls)
        contents = [field(value) for field, value in zip(fields, values)]
        return ODict([(field, content) for field, content in zip(list(cls), contents)])

    @create.register(dict)
    def create_mapping(cls, values, *args, fields, **kwargs):
        values = [values.get(field, None) for field in iter(cls)]
        assert len(values) == len(cls) and None not in values
        contents = [field(value) for field, value in zip(fields, values)]
        return ODict([(field, content) for field, content in zip(list(cls), contents)])


@total_ordering
class Query(ABC, metaclass=QueryMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, contents, *args, delimiter, **kwargs):
        assert isinstance(contents, list) and isinstance(delimiter, str)
        self.__delimiter = str(delimiter)
        self.__contents = tuple(contents)

    def __iter__(self): return iter(self.contents)
    def __hash__(self): return hash(tuple([hash(content) for content in self.contents]))
    def __str__(self): return str(self.delimiter).join([str(content) for content in self.contents])

    def __eq__(self, other):
        assert type(other) is type(self) and list(type(self)) == list(type(other))
        return all([primary == secondary for primary, secondary in zip(self, other)])

    def __lt__(self, other):
        assert type(other) is type(self) and list(type(self)) == list(type(other))
        return all([primary < secondary for primary, secondary in zip(self, other)])

    def items(self): return ODict([(content.name, content.value) for content in self.contents]).items()
    def values(self): return [content.value for content in self.contents]
    def keys(self): return [content.name for content in self.contents]

    @property
    def delimiter(self): return self.__delimiter
    @property
    def contents(self): return self.__contents


