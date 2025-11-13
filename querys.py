# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Query Object
@author: Jack Kirby Cook

"""

import types
import inspect
from enum import Enum
from numbers import Number
from functools import total_ordering
from datetime import date as Date
from datetime import datetime as Datetime
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Query", "Field"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


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
    def parse(string, *args, formatting, **kwargs): return Datetime.strptime(string, formatting)
    @staticmethod
    def encode(value, *args, **kwargs): return hash(Datetime(year=value.year, month=value.month, day=value.day).timestamp())
    @staticmethod
    def string(value, *args, formatting, **kwargs): return str(value.strftime(formatting))

class FieldEnum(FieldData, datatype=Enum, dataparams={"variable": None}):
    @staticmethod
    def parse(string, *args, variable, **kwargs): return variable(string)
    @staticmethod
    def encode(value, *args, variable, **kwargs): return hash(variable(value))
    @staticmethod
    def string(value, *args, variable, **kwargs): return str(variable(value))


@total_ordering
class FieldBase(ABC):
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


class Field(ABCMeta):
    def __new__(mcs, dataname, datatype, **dataparams):
        fielddata = {subclass.datatype: subclass for subclass in FieldData.__subclasses__()}[datatype]
        cls = super(Field, mcs).__new__(mcs, dataname, (fielddata, FieldBase, ABC), {})
        return cls

    def __str__(cls): return str(cls.dataname)
    def __init__(cls, dataname, datatype, **dataparams):
        assert dataname == cls.__name__ and inspect.isclass(datatype) and datatype is cls.datatype
        cls.dataparams = {key: dataparams.get(key, default) for key, default in cls.dataparams.items()}
        cls.datatype = datatype
        cls.dataname = dataname

    def __getitem__(cls, datastring):
        assert isinstance(datastring, str)
        datavalue = cls.parse(datastring, **cls.dataparams)
        assert isinstance(datavalue, cls.datatype)
        instance = super(Field, cls).__call__(cls.dataname, datavalue, cls.dataparams)
        return instance

    def __call__(cls, datavalue):
        assert isinstance(datavalue, cls.datatype)
        instance = super(Field, cls).__call__(cls.dataname, datavalue, cls.dataparams)
        return instance

    @staticmethod
    @abstractmethod
    def parse(string, *args, **kwargs): pass


@total_ordering
class QueryBase(ABC):
    def __init__(self, contents, delimiter):
        assert isinstance(contents, list) and isinstance(delimiter, str)
        self.__delimiter = str(delimiter)
        self.__contents = tuple(contents)

    def __iter__(self): return iter(ODict([(content.name, content.value) for content in self.contents]).items())
    def __hash__(self): return hash(tuple([hash(content) for content in self.contents]))
    def __str__(self): return str(self.delimiter).join([str(content) for content in self.contents])

    def __getattr__(self, attr):
        contents = {content.name: content for content in self.contents}
        if attr not in contents.keys():
            raise AttributeError(attr)
        return contents[attr].value

    def __eq__(self, other):
        if isinstance(other, types.NoneType): return False
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


class Query(ABCMeta):
    def __new__(mcs, name, *args, bases=[], fields=[], **kwargs):
        bases = tuple([QueryBase] + list(bases) + [ABC])
        cls = super(Query, mcs).__new__(mcs, name, bases, {})
        return cls

    def __iter__(cls): return map(str, cls.datafields)
    def __init__(cls, name, *args, fields=[], delimiter="|", **kwargs):
        assert name == cls.__name__ and isinstance(fields, list)
        assert all([isinstance(field, Field) for field in fields])
        cls.datafields = {str(field): field for field in fields}
        cls.dataname = str(name)
        cls.delimiter = delimiter

    def __getitem__(cls, strings):
        assert isinstance(strings, str)
        strings = str(strings).split(cls.delimiter)
        assert len(strings) == len(cls.datafields)
        mapping = ODict([(name, field[string]) for (name, field), string in zip(cls.datafields.items(), strings)])
        contents = list(mapping.values())
        instance = super(Query, cls).__call__(contents, delimiter=cls.delimiter)
        return instance

    def __call__(cls, parameters):
        if isinstance(parameters, str):
            strings = str(parameters).split(cls.delimiter)
            assert len(strings) == len(cls.datafields)
            contents = [field[string] for field, string in zip(cls.datafields.values(), strings)]
        elif isinstance(parameters, list):
            assert len(parameters) == len(cls.datafields)
            contents = [field(value) for field, value in zip(cls.datafields.values(), parameters)]
        elif isinstance(parameters, dict):
            assert len(parameters) >= len(cls.datafields)
            contents = [field(parameters[name]) for name, field in cls.datafields.items()]
        elif isinstance(parameters, QueryBase):
            values = ODict(parameters.items())
            contents = [field(values[name]) for name, field in cls.datafields.items()]
        else: raise TypeError(type(parameters))
        instance = super(Query, cls).__call__(contents, delimiter=cls.delimiter)
        return instance

    @property
    def fields(cls): return cls.datafields





