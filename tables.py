# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, abstractmethod

from support.meta import AttributeMeta
from support.mixins import Fields

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Options", "Table"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class OptionsMeta(AttributeMeta): pass
class Options(Fields, metaclass=OptionsMeta): pass
class DataframeOptions(Options, fields=["rows", "columns", "width", "formats", "numbers"], attribute="Dataframe"): pass


class TableMeta(AttributeMeta):
    def __init__(cls, *args, **kwargs):
        super(TableMeta, cls).__init__(*args, **kwargs)
        cls.__options__ = kwargs.get("options", getattr(cls, "__options__", None))
        cls.__type__ = kwargs.get("type", getattr(cls, "__type__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__options__ is not None
        assert cls.__type__ is not None
        parameters = dict()
        stack = cls.__type__(**parameters)
        mutex = multiprocessing.RLock()
        wrapper = super(TableMeta, cls).__call__(stack, *args, mutex=mutex, **kwargs)
        return wrapper


class Table(ABC, metaclass=TableMeta):
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


class DataframeTable(Table, type=pd.DataFrame, register="Dataframe"):
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
    def string(self): return self.table.to_string(**self.options.todict(), show_dimensions=True)
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)



