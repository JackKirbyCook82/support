# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict


__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Tables", "Views"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ViewMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        order = kwargs.get("order", getattr(cls, "__order__", []))
        fields = getattr(cls, "__fields__", {}) | kwargs.get("fields", {})
        values = getattr(cls, "__values__", {})
        update = {key: kwargs.get(key, values.get(key, None)) for key in fields.keys()}
        values = values | update
        cls.__values__ = values
        cls.__fields__ = fields
        cls.__order__ = order

    def __call__(cls, *args, **kwargs):
        fields = {key: field for key, field in cls.__fields__.items()}
        values = {key: value for key, value in cls.__values__.items()}
        order = list(cls.__order__)
        update = {key: kwargs.get(key, value) for key, value in values.items()}
        values = values | update
        parameters = dict(fields=fields, values=values, order=order)
        instance = super(ViewMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class View(ABC, metaclass=ViewMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, fields={}, values={}, order=[], **kwargs):
        assert set(fields.keys()) == set(values.keys()) and isinstance(order, list)
        self.__border = "=" * values.get("width", 250)
        self.__values = values
        self.__fields = fields
        self.__order = order

    def __call__(self, table, *args, **kwargs):
        table = self.execute(table, *args, **kwargs)
        assert isinstance(table, (str, type(None)))
        string = "\n".join([self.border, table, self.border]) if table is not None else ""
        return string + "\n" if bool(string) else string

    @abstractmethod
    def execute(self, table, *args, **kwargs): pass

    @property
    def parameters(self): return {field: self.values[key] for key, field in self.fields.items()}
    @property
    def border(self): return self.__border
    @property
    def values(self): return self.__values
    @property
    def fields(self): return self.__fields
    @property
    def order(self): return self.__order


class DataframeView(View, fields={"rows": "max_rows", "columns": "max_cols", "width": "line_width", "formats": "formatters", "numbers": "float_format"}):
    def execute(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty):
            return
        columns = self.order if bool(self.order) else list(dataframe.columns)
        length = dataframe.columns.nlevels if isinstance(dataframe.columns, pd.MultiIndex) else 1
        columns = list(self.generator(columns, length)) if length > 1 else list(columns)
        parameters = {key: value for key, value in self.parameters.items()}
        formatters = parameters.pop("formatters", {})
        formatters = zip(self.generator(formatters.keys(), length), formatters.values())
        parameters = parameters | {"formatters": ODict(list(formatters))}
        return dataframe[columns].to_string(**parameters, show_dimensions=True)

    @staticmethod
    def generator(columns, length):
        for column in columns:
            column = tuple([column]) if not isinstance(column, tuple) else column
            column = column + tuple([""]) * (length - len(column))
            yield column if len(column) > 1 else column[0]


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is TableMeta for base in cls.__bases__]):
            return
        cls.__tabletype__ = kwargs.get("tabletype", getattr(cls, "__tabletype__", None))
        cls.__tableview__ = kwargs.get("tableview", getattr(cls, "__tableview__", None))
        cls.__tableaxes__ = kwargs.get("tableaxes", getattr(cls, "__tableaxes__", None))

    def __call__(cls, *args, **kwargs):
        parameters = cls.parameters(table=cls.__tabletype__, view=cls.__tableview__, axes=cls.__tableaxes__)
        instance = super(TableMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __bool__(self): return not self.empty if self.table is not None else False
    def __str__(self): return self.string
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, *args, table, view, axes, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__mutex = multiprocessing.RLock()
        self.__table = table
        self.__view = view
        self.__axes = axes

    def __setitem__(self, locator, value): self.set(locator, value)
    def __getitem__(self, locator): return self.get(locator)

    @classmethod
    @abstractmethod
    def parameters(cls, *args, **kwargs): pass
    @abstractmethod
    def get(self, locator): pass
    @abstractmethod
    def set(self, locator, value): pass

    @property
    @abstractmethod
    def string(self): pass
    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass

    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table
    @property
    def mutex(self): return self.__mutex
    @property
    def view(self): return self.__view
    @property
    def axes(self): return self.__axes
    @property
    def name(self): return self.__name


class DataframeTable(Table, tabletype=pd.DataFrame):
    def get(self, locator):
        index, columns = self.locator(locator)
        return self.table.iloc[index, columns]

    def set(self, locator, value):
        index, columns = self.locator(locator)
        self.table.iloc[index, columns] = value

    def locator(self, locator):
        default = slice(None, None, None)
        locator = locator if isinstance(locator, tuple) else (default, locator)
        index, columns = locator
        assert isinstance(index, slice) and isinstance(columns, (str, list))
        columns = [columns] if isinstance(columns, str) else columns
        columns = [self.stacker(column) for column in columns]
        columns = [list(self.columns).index(value) for value in columns]
        return index, columns

    def stacker(self, column):
        if isinstance(self.columns, pd.MultiIndex):
            column = tuple([column]) if not isinstance(column, tuple) else column
            length = self.columns.nlevels - len(column)
            column = column + tuple([""]) * length
        return column

    def combine(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            self.table = pd.concat([self.dataframe, dataframe], axis=0) if bool(self) else dataframe

    def where(self, function):
        if not bool(self): return
        with self.mutex:
            assert callable(function)
            mask = function(self)
            self.table.where(mask, inplace=True)
            self.table.dropna(how="all", inplace=True)

    def remove(self, function):
        if not bool(self): return
        with self.mutex:
            assert callable(function)
            mask = function(self)
            self.table.where(~mask, inplace=True)
            self.table.dropna(how="all", inplace=True)

    def extract(self, function):
        if not bool(self): return
        with self.mutex:
            assert callable(function)
            mask = function(self)
            dataframe = self.table.where(mask, inplace=False)
            self.table.where(~mask, inplace=True)
            self.table.dropna(how="all", inplace=True)
            return dataframe

    def change(self, function, column, value):
        if not bool(self): return
        with self.mutex:
            assert callable(function)
            column = self.stacker(column)
            mask = function(self)
            self.table.loc[mask, column] = value

    def unique(self, columns, reverse=True):
        if not bool(self): return
        with self.mutex:
            columns = [self.stacker(column) for column in columns]
            keep = ("last" if bool(reverse) else "first")
            parameters = dict(keep=keep, inplace=True)
            self.table.drop_duplicates(columns, **parameters)

    def sort(self, column, reverse=True):
        if not bool(self): return
        with self.mutex:
            column = self.stacker(column)
            ascending = not bool(reverse)
            parameters = dict(ascending=ascending, axis=0, ignore_index=False, inplace=True)
            self.table.sort_values(column, **parameters)

    def reset(self):
        if not bool(self): return
        with self.mutex:
            self.table.reset_index(drop=True, inplace=True)

    @classmethod
    def parameters(cls, *args, table, view, axes, **kwargs):
        view = view(*args, **kwargs)
        axes = axes(*args, **kwargs)
        table = table(columns=axes.header)
        return dict(view=view, axes=axes, table=table)

    @property
    def string(self): return str(self.view(self.table))
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)

    @property
    def dataframe(self): return self.table
    @property
    def index(self): return self.table.index
    @property
    def columns(self): return self.table.columns


class Tables(object):
    Dataframe = DataframeTable


class Views(object):
    Dataframe = DataframeView



