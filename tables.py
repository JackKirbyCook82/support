# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Fields
from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Options", "Tables"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class OptionsMeta(ABCMeta): pass
class Options(Fields, metaclass=OptionsMeta): pass
class DataframeOptions(Options, fields=["rows", "columns", "width", "formats", "numbers"]):
    @property
    def parameters(self):
        return dict(max_rows=self.rows, max_cols=self.columns, line_width=self.width, formatters=self.formats, float_format=self.numbers)


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is TableMeta for base in cls.__bases__]):
            return
        cls.__tabletype__ = kwargs.get("tabletype", getattr(cls, "__tabletype__", None))
        cls.__options__ = kwargs.get("options", getattr(cls, "__options__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__tabletype__ is not None
        assert cls.__options__ is not None
        instance = cls.__tabletype__()
        parameters = dict(mutex=multiprocessing.RLock(), options=cls.__options__)
        instance = super(TableMeta, cls).__call__(instance, *args, **parameters, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __bool__(self): return not self.empty if self.table is not None else False
    def __str__(self): return self.string
    def __len__(self): return self.size

    def __setitem__(self, locator, content): self.set(locator, content)
    def __getitem__(self, locator): return self.get(locator)

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, instance, *args, mutex, options, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__options = options
        self.__table = instance
        self.__mutex = mutex

    @property
    @abstractmethod
    def string(self): pass
    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass

    @abstractmethod
    def set(self, locator, content): pass
    @abstractmethod
    def get(self, locator): pass

    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table

    @property
    def options(self): return self.__options
    @property
    def mutex(self): return self.__mutex
    @property
    def name(self): return self.__name


class DataframeTable(Table, tabletype=pd.DataFrame):
    def set(self, locator, content):
        index, column = self.locator(locator)
        self.table.iloc[index, column] = content

    def get(self, locator):
        index, column = self.locator(locator)
        return self.table.iloc[index, column]

    def locator(self, locator):
        index, columns = locator
        assert isinstance(index, (int, slice))
        assert isinstance(columns, (str, list))
        assert all([isinstance(column, str) for column in columns]) if isinstance(columns, list) else True
        index = self.indexer(index)
        columns = self.stacker(columns)
        numerical = lambda column: list(self.table.columns).index(column)
        columns = [numerical(column) for column in columns] if isinstance(columns, list) else numerical(columns)
        return index, columns

    def indexer(self, index):
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self.table.index)
            index = slice(start, stop, index.step)
        return index

    def stacker(self, columns):
        if isinstance(self.table.columns, pd.MultiIndex):
            group = lambda column: tuple([column] if not isinstance(column, list) else column)
            length = lambda column: self.table.columns.nlevels - len(group(column))
            pad = lambda column: group(column) + tuple([""]) * length(column)
            columns = [pad(column) for column in columns] if isinstance(columns, list) else pad(columns)
        return columns

    def concat(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframe = pd.concat([self.table, dataframe], axis=0) if bool(self) else dataframe
            self.table = dataframe

    def unique(self, columns):
        if not bool(self):
            return
        assert isinstance(columns, list)
        assert all([column in self.columns for column in columns])
        with self.mutex:
            columns = self.stacker(columns)
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
            self.table.drop(dataframe.index, inplace=False)
            return dataframe

    def change(self, function, columns, value):
        if not bool(self):
            return
        assert callable(function) and isinstance(columns, list)
        assert all([column in self.columns for column in columns])
        with self.mutex:
            mask = function(self.table)
            self.table.loc[mask, columns] = value

    def sort(self, column, reverse):
        if not bool(self):
            return
        assert column in self.columns
        with self.mutex:
            self.table.sort_values(column, axis=0, ascending=not bool(reverse), inplace=True, ignore_index=False)

    @property
    def string(self): return self.table.to_string(**self.options.parameters, show_dimensions=True)
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)
    @property
    def columns(self): return self.table.columns
    @property
    def index(self): return self.table.index


class Tables(object):
    Dataframe = DataframeTable


class Options(object):
    Dataframe = DataframeOptions



