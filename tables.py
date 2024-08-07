# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Tables", "Views"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ViewMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        fields = getattr(cls, "__fields__", {}) | kwargs.get("fields", {})
        values = getattr(cls, "__values__", {})
        update = {key: kwargs.get(key, values.get(key, None)) for key in fields.keys()}
        values = values | update
        cls.__values__ = values
        cls.__fields__ = fields

    def __call__(cls, *args, **kwargs):
        fields = {key: field for key, field in cls.__fields__.items()}
        values = {key: value for key, value in cls.__values__.items()}
        update = {key: kwargs.get(key, value) for key, value in values.items()}
        values = values | update
        instance = super(ViewMeta, cls).__call__(*args, fields=fields, values=values, **kwargs)
        return instance


class View(ABC, metaclass=ViewMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, fields={}, values={}, **kwargs):
        assert set(fields.keys()) == set(values.keys())
        self.__border = "-" * values.get("width", 250)
        self.__values = values
        self.__fields = fields

    def __call__(self, *args, **kwargs):
        table = self.table(*args, **kwargs)
        contents = self.contents(*args, **kwargs)
        assert isinstance(table, str) and isinstance(contents, list)
        string = "\n".join([self.border] + [table] + list(contents) + [self.border])
        return string

    @abstractmethod
    def contents(self, *args, **kwargs): pass
    @abstractmethod
    def table(self, *args, **kwargs): pass

    @property
    def parameters(self): return {field: self.values[key] for key, field in self.fields.items()}
    @property
    def values(self): return self.__values
    @property
    def fields(self): return self.__fields
    @property
    def border(self): return self.__border


class DataframeView(View, fields={"rows": "max_rows", "columns": "max_cols", "width": "line_width", "formats": "formatters", "numbers": "float_format"}):
    def contents(*args, **kwargs): return []
    def table(self, *args, table, **kwargs):
        assert isinstance(table, pd.DataFrame)
        return table.to_string(**self.parameters, show_dimensions=True)


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is TableMeta for base in cls.__bases__]):
            return
        cls.__tabletype__ = kwargs.get("tabletype", getattr(cls, "__tabletype__", None))
        cls.__tableview__ = kwargs.get("tableview", getattr(cls, "__tableview__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__tabletype__ is not None
        assert cls.__tableview__ is not None
        instance = cls.__tabletype__()
        view = cls.__tableview__()
        parameters = dict(mutex=multiprocessing.RLock(), view=view)
        instance = super(TableMeta, cls).__call__(instance, *args, **parameters, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __str__(self): return str(self.view(table=self.table))
    def __bool__(self): return not self.empty if self.table is not None else False
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, instance, *args, mutex, view, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__table = instance
        self.__mutex = mutex
        self.__view = view

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
    def name(self): return self.__name


class DataframeTable(Table, tabletype=pd.DataFrame):
    def get(self, index, columns):
        index, columns = self.locator(index, columns)
        return self.table.iloc[index, columns]

    def set(self, index, columns, value):
        index, columns = self.locator(index, columns)
        self.table.iloc[index, columns] = value

    def locator(self, index, columns):
        index = [index] if not isinstance(index, list) else index
        index = [list(self.index).index(value) for value in index]
        columns = [columns] if not isinstance(columns, list) else columns
        columns = [list(self.columns).index(value) for value in columns]
        columns = [self.stacker(column) for column in columns]
        return index, columns

    def stacker(self, column):
        if isinstance(self.table.columns, pd.MultiIndex):
            column = tuple([column]) if not isinstance(column, tuple) else column
            length = self.columns.nlevels - len(column)
            column = column + tuple([""]) * length
        return column

    def concat(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframe = pd.concat([self.table, dataframe], axis=0) if bool(self) else dataframe
            self.table = dataframe

    def unique(self, columns):
        if not bool(self):
            return
        with self.mutex:
            columns = [columns] if not isinstance(columns, list) else columns
            columns = [self.stacker(column) for column in columns]
            self.table.drop_duplicates(columns, keep="last", inplace=True)

    def where(self, function):
        if not bool(self):
            return
        assert callable(function)
        with self.mutex:
            mask = function(self.table)
            self.table.where(mask).dropna(how="all", inplace=True)

    def remove(self, function):
        if not bool(self):
            return
        assert callable(function)
        with self.mutex:
            mask = function(self.table)
            dataframe = self.table.where(mask)
            dataframe = dataframe.dropna(how="all", inplace=False)
            self.table.drop(dataframe.index, inplace=True)
            return dataframe

    def change(self, function, columns, value):
        if not bool(self):
            return
        assert callable(function)
        with self.mutex:
            columns = [columns] if not isinstance(columns, list) else columns
            columns = [self.stacker(column) for column in columns]
            mask = function(self.table)
            self.table.loc[mask, columns] = value

    def sort(self, column, reverse):
        if not bool(self):
            return
        with self.mutex:
            column = self.stacker(column)
            ascending = not bool(reverse)
            self.table.sort_values(column, axis=0, ascending=ascending, inplace=True, ignore_index=False)

    def reset(self):
        with self.mutex:
            self.table.reset_index(drop=True, inplace=True)

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


class Tables(object):
    Dataframe = DataframeTable


class Views(object):
    Dataframe = DataframeView



