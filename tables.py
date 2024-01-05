# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from support.locks import Locks
from support.pipelines import Stack

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class Table(Stack, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__type__ = kwargs.get("type", getattr(cls, "__type__", None))

    def __len__(self): return self.size
    def __bool__(self): return not self.empty
    def __init__(self, *args, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        tabletype = self.__class__.__tabletype__
        name = str(self.name).replace("Table", "Lock")
        assert tabletype is not None
        self.__mutex = Locks(name=name, timeout=timeout)
        self.__type = tabletype
        self.__tables = dict()

    def read(self, *args, table, **kwargs):
        with self.mutex[str(table)]:
            return self.tables[table]

    def write(self, other, *args, table, **kwargs):
        assert isinstance(other, self.type)
        with self.mutex[str(table)]:
            content = self.tables.get(table, None)
            content = self.create(content, *args, table=table, **kwargs)
            content = self.combine(content, other, *args, table=table, **kwargs)
            content = self.parser(content, *args, table=table, **kwargs)
            content = self.format(content, *args, table=table, **kwargs)
            self.tables[table] = content

    @staticmethod
    @abstractmethod
    def create(content, *args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def combine(content, other, *args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def parser(content, *args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def format(content, *args, **kwargs): pass

    @property
    def tables(self): return self.__tables
    @property
    def mutex(self): return self.__mutex
    @property
    def type(self): return self.__type


class DataframeTable(Table, ABC, type=pd.DataFrame):
    def create(self, dataframe, *args, table, **kwargs):
        header = self.header(*args, table=table, **kwargs)
        dataframe = dataframe if dataframe is not None else pd.DataFrame(columns=header)
        return dataframe

    @staticmethod
    def combine(dataframe, other, *args, **kwargs): return pd.concat([dataframe, other], axis=0)
    @staticmethod
    def parser(dataframe, *args, **kwargs): return dataframe.reset_index(drop=True, inplace=False)
    @staticmethod
    def format(dataframe, *args, **kwargs): return dataframe

    @staticmethod
    @abstractmethod
    def header(*args, **kwargs): pass

    @property
    def size(self): return len(self.table.index)
    @property
    def empty(self): return bool(self.table.empty)







