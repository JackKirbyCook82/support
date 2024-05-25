# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Options", "Table"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


# class OptionsMeta(RegistryMeta):
#     def __new__(mcs, name, bases, attrs, *args, **kwargs):
#         cls = super(OptionsMeta, mcs).__new__(mcs, name, bases, attrs)
#         return cls
#
#     def __call__(cls, *args, **kwargs):
#         fields = [kwargs[field] for field in cls._fields]
#         instance = super(OptionsMeta, cls).__call__(*fields)
#         return instance


# class Options(ABC, metaclass=OptionsMeta):
#     pass


class TableMeta(RegistryMeta):
    def __init__(cls, *args, **kwargs):
        super(TableMeta, cls).__init__(*args, **kwargs)
        cls.Options = kwargs.get("options", getattr(cls, "Options", None))
        cls.Type = kwargs.get("type", getattr(cls, "Type", None))

    def __call__(cls, *args, **kwargs):
        assert cls.Options is not None
        assert cls.Type is not None
        instance = cls.Type()
        parameters = dict(options=cls.Options, mutex=multiprocessing.RLock())
        wrapper = super(TableMeta, cls).__call__(instance, *args, **parameters, **kwargs)
        return wrapper


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __bool__(self): return not self.empty if self.table is not None else False
    def __len__(self): return self.size
    def __str__(self): return self.string

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, instance, *args, options, mutex, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__options = options
        self.__table = instance
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


# class DataframeOptions(ntuple("Options", "rows columns width formats numbers"), Options, key="Dataframe"):
#     def __new__(cls, *args, **kwargs): return super().__new__(cls, *[kwargs[field] for field in cls._fields])
#
#     @property
#     def parameters(self): return dict(max_rows=self.rows, max_cols=self.columns, line_width=self.width, float_format=self.numbers, formatters=self.formats)


class DataframeTable(Table, type=pd.DataFrame, key="Dataframe"):
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



