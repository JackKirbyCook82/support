# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple

from support.mixins import Pipelining, Sourcing, Logging, Emptying, Sizing
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Table", "View"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ViewField(ntuple("Field", "attr key value")):
    def __call__(self, value): return type(self)(self.attr, self.key, value)


class ViewMeta(RegistryMeta, ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is ViewMeta for base in bases]):
            return super(ViewMeta, mcs).__new__(mcs, name, bases, attrs)
        datatype = kwargs.get("datatype", None)
        if datatype is not None: bases = tuple([View[datatype]] + list(bases))
        exclude = [key for key, value in attrs.items() if isinstance(value, ViewField)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(ViewMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(ViewMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([type(base) is ViewMeta for base in cls.__bases__]):
            return
        fields = {key: value for key, value in attrs.items() if isinstance(value, ViewField)}
        cls.__fields__ = getattr(cls, "__fields__", {}) | fields
        cls.__order__ = kwargs.get("order", getattr(cls, "__order__", []))

    def __call__(cls, *args, **kwargs):
        assert "order" not in kwargs.keys() and "fields" not in kwargs.keys()
        fields, order = list(cls.__fields__.values()), list(cls.__order__)
        fields = {attr: ViewField(attr, key, kwargs.get(attr, value)) for (attr, key, value) in fields}
        parameters = dict(order=order, fields=fields)
        return super(ViewMeta, cls).__call__(*args, **parameters, **kwargs)


class View(ABC, metaclass=ViewMeta):
    def __init__(self, *args, order=[], fields=[], **kwargs):
        self.__fields = dict(fields)
        self.__order = list(order)

    def __call__(self, table, *args, **kwargs):
        table = self.execute(table, *args, **kwargs)
        assert isinstance(table, (str, type(None)))
        string = "\n".join(["=" * 250, table, "=" * 250]) if table is not None else ""
        return string + "\n" if bool(string) else string

    @abstractmethod
    def execute(self, table, *args, **kwargs): pass

    @property
    def fields(self): return self.__fields
    @property
    def order(self): return self.__order


class ViewDataframe(View, ABC, register=pd.DataFrame):
    numbers = ViewField("numbers", "float_format", lambda column: f"{column:.02f}")
    formats = ViewField("formats", "formatters", {})
    columns = ViewField("columns", "max_cols", 30)
    width = ViewField("width", "line_width", 250)
    rows = ViewField("rows", "max_rows", 30)

    def execute(self, table, *args, **kwargs):
        if not bool(table): return
        overlaps = lambda label, column: label in column if isinstance(column, tuple) else label is column
        order = [column for label in self.order for column in table.columns if overlaps(label, column)]
        formats = self.fields.get("formats", {}).items()
        formats = {column: function for column in order for label, function in formats if overlaps(label, column)}
        assert len(list(formats.keys())) == len(set(formats.keys()))
        fields = self.fields | {"formats": self.fields["formats"](formats)}
        parameters = {key: value for (attr, key, value) in fields.values()}
        string = table.dataframe[order].to_string(**parameters, show_dimensions=True)
        return string


class TableData(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.datatype = kwargs.get("datatype", getattr(cls, "datatype", None))

    def __init__(self, *args, table, view, mutex, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mutex = mutex
        self.__table = table
        self.__view = view

    def __bool__(self): return not self.empty if self.table is not None else False
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"
    def __len__(self): return self.size if bool(self) else 0
    def __str__(self): return self.view(self.table)

    def __setitem__(self, locator, value): self.set(locator, value)
    def __getitem__(self, locator): return self.get(locator)

    @abstractmethod
    def set(self, locator, value): pass
    @abstractmethod
    def get(self, locator): pass

    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table
    @property
    def mutex(self): return self.__mutex
    @property
    def view(self): return self.__view

    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass


class TableDataFrame(TableData, datatype=pd.DataFrame):
    def get(self, locator):
        index, columns = self.locator(locator)
        return self.table.iloc[index, columns]

    def set(self, locator, value):
        index, columns = self.locator(locator)
        self.table.iloc[index, columns] = value

    def locate(self, locator):
        index, columns = locator if isinstance(locator, tuple) else (slice(None, None, None), locator)
        assert isinstance(index, slice) and isinstance(columns, (str, list))
        if isinstance(columns, list):
            columns = list(map(self.stack, columns))
            columns = [list(self.columns).index(column) for column in columns]
        else:
            columns = self.stack(columns)
            columns = list(self.columns).index(columns)
        return index, columns

    def stack(self, column):
        if not bool(self.stacked): return column
        column = tuple([column]) if not isinstance(column, tuple) else column
        return column + tuple([]) * (int(self.stacking) - len(column))

    def combine(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            self.table = pd.concat([self.dataframe, dataframe], axis=0) if bool(self) else dataframe

    def where(self, function):
        if not bool(self): return
        assert callable(function)
        with self.mutex:
            mask = function(self)
            self.table.where(mask, inplace=True)
            self.table.dropna(how="all", inplace=True)

    def remove(self, function):
        if not bool(self): return
        assert callable(function)
        with self.mutex:
            mask = function(self)
            self.table.where(~mask, inplace=True)
            self.table.dropna(how="all", inplace=True)

    def extract(self, function):
        if not bool(self): return
        assert callable(function)
        with self.mutex:
            mask = function(self)
            dataframe = self.table.where(mask, inplace=False)
            dataframe = dataframe.dropna(how="all", inplace=False)
            self.table.where(~mask, inplace=True)
            self.table.dropna(how="all", inplace=True)
            return dataframe

    def change(self, function, column, value):
        if not bool(self): return
        assert callable(function) and isinstance(column, str)
        with self.mutex:
            column = self.stack(column)
            mask = function(self)
            self.table.loc[mask, column] = value

    def unique(self, columns, reverse=True):
        if not bool(self): return
        assert isinstance(columns, list)
        with self.mutex:
            columns = [self.stack(column) for column in columns]
            keep = ("last" if bool(reverse) else "first")
            parameters = dict(keep=keep, inplace=True)
            self.table.drop_duplicates(columns, **parameters)

    def sort(self, columns, reverse):
        if not bool(self): return
        assert isinstance(columns, (list, str)) and isinstance(reverse, (list, bool))
        columns = [columns] if isinstance(columns, str) else columns
        reverse = [reverse] if isinstance(reverse, bool) else reverse
        reverse = reverse * len(columns) if len(reverse) == 1 else reverse
        assert len(columns) == len(reverse)
        with self.mutex:
            columns = [self.stack(column) for column in columns]
            ascending = [not bool(value) for value in reverse]
            parameters = dict(ascending=ascending, axis=0, ignore_index=False, inplace=True)
            self.table.sort_values(columns, **parameters)

    def reset(self):
        if not bool(self): return
        with self.mutex:
            self.table.reset_index(drop=True, inplace=True)

    def clear(self):
        if not bool(self): return
        with self.mutex:
            index = self.index
            self.table.drop(index, inplace=True)

#    @staticmethod
#    def parameters(*args, viewtype, datatype, **kwargs):
#        view = viewtype(*args, **kwargs)
#        table = datatype(columns=)
#        return dict(table=table, view=view)

    @property
    def stacking(self): return len(self.columns.nlevels) if bool(self.stacked) else 0
    @property
    def stacked(self): return isinstance(self.columns, pd.MultiIndex)
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)
    @property
    def columns(self): return self.table.columns
    @property
    def index(self): return self.table.index
    @property
    def dataframe(self): return self.table


class TableMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        mixins = {subclass.datatype: subclass for subclass in TableData.__subclasses__()}
        datatype = kwargs.get("datatype", None)
        if datatype is not None: bases = tuple([mixins[datatype]] + list(bases))
        cls = super(TableMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, **kwargs):
        if not any([type(base) is TableMeta for base in cls.__bases__]):
            return
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__viewtype__ = kwargs.get("viewtype", getattr(cls, "__viewtype__", None))

#    def __call__(cls, *args, **kwargs):
#        parameters = dict(variable=cls.__variable__) | cls.parameters(*args, viewtype=cls.__viewtype__, datatype=cls.__datatype__, **kwargs)
#        instance = super(TableMeta, cls).__call__(*args, **parameters, mutex=multiprocessing.RLock(), **kwargs)
#        return instance


class Table(Logging, ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, variable, **kwargs):
        Logging.__init__(self, *args, **kwargs)
        self.__variable = variable

#    @staticmethod
#    @abstractmethod
#    def parameters(*args, **kwargs): pass
    @property
    def variable(self): return self.__variable


class Reader(Pipelining, Logging, Sizing, Emptying):
    def __init__(self, *args, table, **kwargs):
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__table = table

    def execute(self, *args, **kwargs):
        pass

    @property
    def table(self): return self.__table


class Writer(Pipelining, Sourcing, Logging, Sizing, Emptying):
    def __init__(self, *args, table, **kwargs):
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__table = table

    def execute(self, *args, **kwargs):
        pass

    @property
    def table(self): return self.__table









