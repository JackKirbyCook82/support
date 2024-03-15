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
    @table.setter
    def table(self, table): self.__table = table
    @property
    def mutex(self): return self.__mutex
    @property
    def type(self): return self.__type
    @property
    def name(self): return self.__name


class DataframeTable(Table, ABC, type=pd.DataFrame):
    def __init_subclass__(cls, *args, header={}, **kwargs):
        assert isinstance(header, dict)
        cls.header = header

    def __init__(self, *args, **kwargs):
        table = pd.DataFrame(columns=self.header)
        super().__init__(table, *args, **kwargs)

    def read(self, *args, **kwargs):
        with self.mutex:
            return self.table

    def write(self, dataframe, args, **kwargs):
        with self.mutex:
            self.table = pd.concat([self.table, dataframe], axis=0)

    def update(self, dataframe, *args, **kwargs):
        with self.mutex:
            self.table.update(dataframe)

    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)



