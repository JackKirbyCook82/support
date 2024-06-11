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
        stack = cls.__tabletype__()
        mutex = multiprocessing.RLock()
        instance = super(TableMeta, cls).__call__(stack, *args, mutex=mutex, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __bool__(self): return not self.empty if self.table is not None else False
    def __len__(self): return self.size
    def __str__(self): return self.string

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, stack, *args, mutex, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__options = self.__class__.__options__
        self.__table = stack
        self.__mutex = mutex

    def __setitem__(self, locator, content): self.write(locator, content)
    def __getitem__(self, locator): return self.read(locator)

    @abstractmethod
    def write(self, locator, content, *args, **kwargs): pass
    @abstractmethod
    def read(self, locator, *args, **kwargs): pass
    @abstractmethod
    def remove(self, content, *args, **kwargs): pass
    @abstractmethod
    def concat(self, content, *args, **kwargs): pass
    @abstractmethod
    def update(self, content, *args, **kwargs): pass

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
    def options(self): return self.__options
    @property
    def mutex(self): return self.__mutex
    @property
    def name(self): return self.__name


class DataframeTable(Table, tabletype=pd.DataFrame):
    def write(self, locator, content, *args, **kwargs):
        index, column = locator
        assert isinstance(index, (int, slice)) and isinstance(column, int)
        self.table.iloc[index, column] = content

    def read(self, locator, **kwargs):
        index, column = locator
        assert isinstance(index, (int, slice)) and isinstance(column, int)
        return self.table.iloc[index, column]

    def remove(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            self.table = self.table.drop(dataframe.index, inplace=False)

    def concat(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframes = [self.table, dataframe]
            self.table = pd.concat(dataframes, axis=0)

    def update(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            self.table.update(dataframe)

    def sort(self, column, *args, reverse=False, **kwargs):
        assert isinstance(column, str) and isinstance(reverse, bool)
        assert column in self.table.columns
        with self.mutex:
            self.table.sort_values(column, axis=0, ascending=not bool(reverse), inplace=True, ignore_index=False)

    def truncate(self, rows):
        assert isinstance(rows, int)
        with self.mutex:
            self.table = self.table.head(rows)

    @property
    def string(self): return self.table.to_string(**self.options.parameters, show_dimensions=True)
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)


class Tables(object):
    Dataframe = DataframeTable


class Options(object):
    Dataframe = DataframeOptions



