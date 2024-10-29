# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import inspect
import multiprocessing
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.mixins import Function, Generator, Logging, Emptying, Sizing
from support.dispatchers import typedispatcher
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Reader", "Writer", "Table", "View"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ViewField(ntuple("Field", "attribute key value")):
    def __call__(self, value): return type(self)(self.attribute, self.key, value)


class ViewMeta(RegistryMeta):
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
        order = kwargs.get("order", getattr(cls, "__order__", []))
        fields = {key: value for key, value in attrs.items() if isinstance(value, ViewField)}
        fields = getattr(cls, "__fields__", {}) | fields
        update = {attribute: field(kwargs[attribute]) for attribute, field in fields.items() if attribute in kwargs.keys()}
        cls.__fields__ = fields | update
        cls.__order__ = order

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

    def execute(self, dataframe, *args, **kwargs):
        if bool(dataframe.empty): return
        overlaps = lambda label, column: label in column if isinstance(column, tuple) else label is column
        order = [column for label in self.order for column in dataframe.columns if overlaps(label, column)]
        formats = self.fields.get("formats", {}).value.items()
        formats = {column: function for column in order for label, function in formats if overlaps(label, column)}
        assert len(list(formats.keys())) == len(set(formats.keys()))
        fields = self.fields | {"formats": self.fields["formats"](formats)}
        parameters = {key: value for (attr, key, value) in fields.values()}
        string = dataframe[order].to_string(**parameters, show_dimensions=True)
        return string


class TableMeta(RegistryMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is TableMeta for base in bases]):
            return super(TableMeta, mcs).__new__(mcs, name, bases, attrs)
        datatype = kwargs.get("datatype", None)
        if datatype is not None: bases = tuple([Table[datatype]] + list(bases))
        cls = super(TableMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, **kwargs):
        super(TableMeta, cls).__init__(*args, **kwargs)
        cls.datatype = kwargs.get("datatype", getattr(cls, "datatype", None))

    def __call__(cls, *args, name=None, **kwargs):
        parameters = dict(name=cls.__name__)
        instance = super(TableMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init__(self, *args, table, view, name, **kwargs):
        assert isinstance(table, type(self).datatype) and isinstance(view, View)
        self.__mutex = multiprocessing.RLock()
        self.__table = table
        self.__view = view
        self.__name = name

    def __bool__(self): return not bool(self.empty) if self.table is not None else False
    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __len__(self): return int(self.size) if bool(self) else 0
    def __str__(self): return str(self.view(self.table))

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
    def name(self): return self.__name

    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass


class TableDataFrame(Table, register=pd.DataFrame):
    def get(self, locator):
        index, columns = locator
        assert isinstance(index, slice) and isinstance(columns, (str, list))
        columns = self.locate(columns)
        return self.table.iloc[index, columns]

    def set(self, locator, value):
        index, columns = locator
        assert isinstance(index, slice) and isinstance(columns, (str, list))
        columns = self.locate(columns)
        self.table.iloc[index, columns] = value

    @typedispatcher
    def locate(self, columns): raise TypeError(type(columns))
    @locate.register(list)
    def locate_multiple(self, columns): return [self.locate(column) for column in columns]
    @locate.register(str)
    def locate_single(self, column):
        column = self.stack(column)
        column = list(self.columns).index(column)
        return column

    def stack(self, column):
        if not bool(self.stacked): return column
        column = tuple([column]) if not isinstance(column, tuple) else column
        return column + tuple([""]) * (int(self.stacking) - len(column))

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

    @property
    def stacking(self): return int(self.columns.nlevels) if bool(self.stacked) else 0
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


class Writer(Function, Logging, Sizing, Emptying, ABC):
    def __init__(self, *args, table, **kwargs):
        assert not inspect.isgeneratorfunction(self.write)
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__table = table

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        query, content = source
        if self.empty(content): return
        with self.table.mutex:
            self.write(query, content, *args, **kwargs)
            size = self.size(content)
            string = f"Wrote: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)

    @abstractmethod
    def write(self, *args, **kwargs): pass

    @property
    def table(self): return self.__table


class Reader(Generator, Logging, Sizing, Emptying, ABC):
    def __init__(self, *args, table, query, **kwargs):
        assert not inspect.isgeneratorfunction(self.read)
        Generator.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__query = query
        self.__table = table

    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            contents = self.read(*args, **kwargs)
            if self.empty(contents): return
            for query, content in self.source(contents):
                size = self.size(content)
                string = f"Read: {repr(self)}|{query}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(content): continue
                yield query, content

    @typedispatcher
    def source(self, contents): raise TypeError(type(contents))

    @source.register(pd.DataFrame)
    def source_dataframe(self, dataframe):
        generator = dataframe.groupby(list(self.query))
        for values, dataframe in iter(generator):
            query = self.query(values)
            yield query, dataframe

    @source.register(xr.Dataset)
    def source_dataset(self, dataset):
        for field in list(self.query):
            dataset = dataset.expand_dims(field)
        dataset = dataset.stack(stack=list(self.query))
        generator = dataset.groupby("stack")
        for values, dataset in iter(generator):
            query = self.query(values)
            dataset = dataset.unstack().drop_vars("stack")
            yield query, dataset

    @abstractmethod
    def read(self, *args, **kwargs): pass

    @property
    def query(self): return self.__query
    @property
    def table(self): return self.__table



