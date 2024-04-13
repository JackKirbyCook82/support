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

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Tabulation", "Tables", "Options"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.TableType = kwargs.get("type", getattr(cls, "TableType", None))

    def __call__(cls, *args, **kwargs):
        assert cls.TableType is not None
        instance = cls.TableType()
        instance = super(TableMeta, cls).__call__(instance, *args, table=instance, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __bool__(self): return not self.empty if self.table is not None else False
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __str__(self): return self.string
    def __init__(self, instance, *args, options, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__mutex = multiprocessing.RLock()
        self.__options = options
        self.__table = instance

    def __setitem__(self, locator, content): self.put(locator, content)
    def __getitem__(self, locator): return self.get(locator)

    @abstractmethod
    def put(self, locator, content, *args, **kwargs): pass
    @abstractmethod
    def get(self, locator, *args, **kwargs): pass
    @abstractmethod
    def remove(self, content, *args, **kwargs): pass
    @abstractmethod
    def concat(self, content, *args, **kwargs): pass
    @abstractmethod
    def update(self, content, *args, **kwargs): pass

    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass
    @property
    @abstractmethod
    def string(self): pass

    @property
    def name(self): return self.__name
    @property
    def mutex(self): return self.__mutex
    @property
    def options(self): return self.__options
    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table


class DataframeOptions(ntuple("Options", "rows columns width format")):
    def __new__(cls, *args, **kwargs): return super().__new__(cls, *[kwargs[field] for field in cls._fields])

class DataframeTable(Table, ABC, type=pd.DataFrame):
    def put(self, locator, content, *args, **kwargs): self.table.loc[locator] = content
    def get(self, locator, **kwargs): return self.table.loc[locator]

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

    @property
    def parameters(self): return dict(max_rows=self.options.rows, max_cols=self.options.columns, line_width=self.options.width, float_format=self.options.format)
    @property
    def string(self): return self.table.to_string(**self.parameters, show_dimensions=True)
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)


class Tabulation(ABC):
    def __repr__(self): return f"{self.name}[{', '.join([variable for variable in self.files.keys()])}]"
    def __getitem__(self, variable): return self.tables[variable]
    def __init__(self, *args, tables=[], **kwargs):
        assert isinstance(tables, list)
        assert all([isinstance(instance, Table) for instance in tables])
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__tables = {str(instance.variable): instance for instance in tables}

    @property
    def tables(self): return self.__tables
    @property
    def name(self): return self.__name


class Tables:
    Dataframe = DataframeTable

class Options:
    Dataframe = DataframeOptions



