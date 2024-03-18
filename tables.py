# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.locks import Lock

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__type__ = kwargs.get("type", getattr(cls, "__type__", None))

    def __call__(cls, *args, **kwargs):
        filename = kwargs.get("name", cls.__name__)
        filetype = cls.__type__
        assert filetype is not None
        instance = super(TableMeta, cls).__call__(filename, filetype, *args, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __bool__(self): return not self.empty if self.table is not None else False
    def __len__(self): return self.size

    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, table, tablename, tabletype, *args, timeout=None, **kwargs):
        lockname = str(tablename).replace("Table", "Lock")
        self.__mutex = Lock(name=lockname, timeout=timeout)
        self.__type = tabletype
        self.__name = tablename
        self.__table = table

    @abstractmethod
    def write(self, content, *args, **kwargs): pass
    @abstractmethod
    def update(self, content, *args, **kwargs): pass
    @abstractmethod
    def read(self, *args, **kwargs): pass

    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass

    @property
    def table(self): return self.__table
    @property
    def mutex(self): return self.__mutex
    @property
    def type(self): return self.__type
    @property
    def name(self): return self.__name


class DataframeTable(Table, ABC, type=pd.DataFrame):
    def __init__(self, *args, **kwargs):
        table = pd.DataFrame()
        super().__init__(table, *args, **kwargs)

    def read(self, *args, **kwargs):
        with self.mutex:
            return self.table

    def remove(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            self.table = self.table.drop(dataframe.index, inplace=False)

    def write(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframes = [self.table, dataframe]
            self.table = pd.concat(dataframes, axis=0)

    def update(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            self.table.update(dataframe)

    def sort(self, column, *args, reverse=False, **kwargs):
        assert column in self.table.columns
        assert isinstance(reverse, bool)
        with self.mutex:
            ascending = not bool(reverse)
            self.table.sort_values(column, axis=0, ascending=ascending, inplace=True, ignore_index=False)

    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)




