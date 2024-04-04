# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.TableType = kwargs.get("type", getattr(cls, "TableType", None))

    def __call__(cls, *args, **kwargs):
        assert cls.Table is not None
        instance = cls.TableType()
        instance = super(TableMeta, cls).__call__(instance, *args, table=instance, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return self.__class__.__name__
    def __bool__(self): return not self.empty if self.table is not None else False
    def __len__(self): return self.size
    def __init__(self, instance, *args, **kwargs):
        self.__mutex = multiprocessing.RLock()
        self.__table = instance

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
    def table(self): return self.__table
    @property
    def mutex(self): return self.__mutex


class DataframeTable(Table, ABC, type=pd.DataFrame):
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
        assert column in self.table.columns
        assert isinstance(reverse, bool)
        with self.mutex:
            ascending = not bool(reverse)
            self.table.sort_values(column, axis=0, ascending=ascending, inplace=True, ignore_index=False)

    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)




