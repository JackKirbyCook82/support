# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.locks import Lock

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


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
    def execute(self, content, *args, **kwargs): pass
    @abstractmethod
    def write(self, content, *args, **kwargs): pass
    @abstractmethod
    def read(self, *args, **kwargs): pass

    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table
    @property
    def mutex(self): return self.__mutex
    @property
    def type(self): return self.__type
    @property
    def name(self): return self.__name


class DataframeTable(Table, ABC, type=pd.DataFrame):
    def __init__(self, *args, **kwargs): super().__init__(pd.DataFrame(columns=self.header), *args, **kwargs)
    def __iter__(self): return (self.parser(index, record) for index, record in self.table.to_dict("index").items())

    def read(self, *args, include=[], exclude=[], **kwargs):
        with self.mutex:
            mask = self.table.index.isin(include) if bool(include) else ~self.table.index.isin(exclude)
            dataframe = self.table[mask]
            return dataframe

    def remove(self, *args, include=[], exclude=[], **kwargs):
        with self.mutex:
            mask = self.table.index.isin(include) if bool(include) else ~self.table.index.isin(exclude)
            dataframe = self.table[mask]
            self.table = self.table[~mask]
            return dataframe

    def write(self, dataframe, args, **kwargs):
        with self.mutex:
            self.table = pd.concat([self.table, dataframe], axis=0)
            self.execute(*args, **kwargs)

    def update(self, dataframe, *args, **kwargs):
        with self.mutex:
            self.table.update(dataframe)
            self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        pass

    @staticmethod
    def included(index, include=[], exclude=[]): return index in include if bool(include) else (index not in exclude)
    @staticmethod
    def parser(index, record): return record

    @property
    def index(self): return self.table.index.values
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)

    @property
    @abstractmethod
    def header(self): pass



