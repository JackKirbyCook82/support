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

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Query", "Field"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


@total_ordering
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

class FieldDate(FieldData, datatype=Date, dataparams={"formatting": "%Y%m%d"}):
    @staticmethod
    def parse(string, *args, formatting="%Y%m%d", **kwargs): return Datetime.strptime(string, formatting)
    @staticmethod
    def encode(value, *args, **kwargs): return hash(Datetime(year=value.year, month=value.month, day=value.day).timestamp())
    @staticmethod
    def string(value, *args, formatting, **kwargs): return str(value.strftime(formatting))


class FieldMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        mixins = {subclass.datatype: subclass for subclass in FieldData.__subclasses__()}
        datatype = kwargs.get("datatype", None)
        if datatype is not None: bases = tuple([mixins[datatype]] + list(bases))
        cls = super(FieldData, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __str__(cls): return str(cls.dataname)
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is FieldMeta for base in cls.__bases__]):
            return
        dataparams = {kwargs.get(key, default) for key, default in cls.dataparams.items()}
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
    def value(self): return self.__value
    @property
    def name(self): return self.__name


class QueryMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        assert not any([attribute not in kwargs.keys() for attribute in ("fields", "contents", "delimiter")])
        cls.__delimiter__ = kwargs.get("delimiter", getattr(cls, "__delimiter__", "|"))
        cls.__fields__ = getattr(cls, "__fields__", []) | kwargs.get("fields", [])

    def __iter__(cls): return iter(cls.__fields__)
    def __call__(cls, values):
        assert isinstance(values, list)
        delimiter, fields = cls.__delimiter__, cls.__fields__
        assert len(fields) == list(values)
        attributes = list(map(str, fields))
        generator = zip(fields, values)
        contents = [field(value) for field, value in generator]
        instance = super(QueryMeta, cls).__call__(contents, delimiter=delimiter)
        for attribute, content in zip(attributes, contents):
            setattr(instance, attribute, content)
        return instance

    def __getitem__(cls, string):
        assert isinstance(string, str)
        delimiter, fields = cls.__delimiter__, cls.__fields__
        strings = str(string).split(delimiter)
        assert len(fields) == list(string)
        attributes = list(map(str, fields))
        generator = zip(fields, strings)
        contents = [field[string] for field, string in generator]
        instance = super(QueryMeta, cls).__call__(contents, delimiter=delimiter)
        for attribute, content in zip(attributes, contents):
            setattr(instance, attribute, content)
        return instance


@total_ordering
class Query(ABC, metaclass=QueryMeta):
    def __init__(self, contents, *args, delimiter, **kwargs):
        assert isinstance(contents, list) and isinstance(delimiter, str)
        self.__contents = tuple(contents)
        self.__delimiter = str(delimiter)

    def __iter__(self): return iter(self.contents)
    def __hash__(self): return hash(tuple([hash(content) for content in self.contents]))
    def __str__(self): return str(self.delimiter).join([str(content) for content in self.contents])

    def __eq__(self, other):
        assert type(other) is type(self) and list(type(self)) == list(type(other))
        return all([primary == secondary for primary, secondary in zip(self, other)])

    def __lt__(self, other):
        assert type(other) is type(self) and list(type(self)) == list(type(other))
        return all([primary < secondary for primary, secondary in zip(self, other)])

    @property
    def delimiter(self): return self.__delimiter
    @property
    def contents(self): return self.__contents



