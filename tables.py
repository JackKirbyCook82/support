# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from support.locks import Lock
from support.pipelines import Stack

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class Table(Stack, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        tabletype = kwargs.get("tabletype", getattr(cls, "__tabletype__", None))
        cls.__tabletype__ = tabletype

    def __len__(self): return self.size
    def __bool__(self): return not self.empty
    def __init__(self, *args, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        tabletype = self.__class__.__tabletype__
        name = str(self.name).replace("Table", "Lock")
        assert tabletype is not None
        self.__mutex = Lock(id(self), name=name, timeout=timeout)
        self.__table = tabletype()

    def read(self, *args, **kwargs):
        with self.mutex:
            return self.table

    def write(self, other, *args, **kwargs):
        table, other = self.table, other[self.header]
        with self.mutex:
            table = self.combine(table, other, *args, **kwargs)
            table = self.parser(table, *args, **kwargs)
            table = self.format(table, *args, **kwargs)
            self.table = table

    @abstractmethod
    def combine(self, table, *args, **kwargs): pass
    @abstractmethod
    def parser(self, table, *args, **kwargs): pass
    @abstractmethod
    def format(self, table, *args, **kwargs): pass

    @property
    def mutex(self): return self.__mutex
    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table


class DataframeTable(Table, ABC, tabletype=pd.DataFrame):
    @staticmethod
    def combine(dataframe, other, *args, **kwargs): return pd.concat([dataframe, other], axis=0)
    @staticmethod
    def parser(dataframe, *args, **kwargs): return dataframe.reset_index(drop=True, inplace=False)
    @staticmethod
    def format(dataframe, *args, **kwargs): return dataframe

    @property
    def size(self): return len(self.table.index)
    @property
    def empty(self): return bool(self.table.empty)

    @property
    @abstractmethod
    def header(self): pass




