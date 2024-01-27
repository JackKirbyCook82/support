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
from support.dispatchers import typedispatcher, valuedispatcher

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
    def read(self, *args, **kwargs): pass
    @abstractmethod
    def write(self, content, *args, **kwargs): pass
    @abstractmethod
    def execute(self, content, *args, **kwargs): pass

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
    def __contains__(self, index): return index in self.table.index

    def generator(self, include=[], exclude=[]): return ((index, record) for index, record in self.table.to_dict("index").items() if self.included(index, include, exclude))
    def dataframe(self, include=[], exclude=[]): return self.table[self.table.index.isin(include) if bool(include) else ~self.table.index.isin(exclude)]
    def records(self, include=[], exclude=[]): return [(index, record) for (index, record) in self.generator(include=include, exclude=exclude)]
    def content(self, index): return {key: record for key, record in self.generator()}[index]

    def execute(self, dataframe, *args, **kwargs):
        start = self.table.index.max() + 1 if not bool(self.table.empty) else 0
        index = np.arange(start, start + len(dataframe.index))
        dataframe = dataframe.set_index(index, drop=True, inplace=False)[self.header]
        return pd.concat([self.table, dataframe], axis=0)

    @staticmethod
    def included(index, include=[], exclude=[]): return index in include if bool(include) else (index not in exclude)

    @valuedispatcher
    def read(self, astype, *args, **kwargs): raise TypeError(astype.__name__)
    @typedispatcher
    def write(self, content, *args, **kwargs): raise TypeError(type(content).__name__)

    @read.register(pd.DataFrame)
    def read_dataframe(self, *args, include=[], exclude=[], **kwargs): return self.select(include, exclude)
    @read.register(list)
    def read_records(self, *args, include=[], exclude=[], **kwargs): return self.records(include, exclude)
    @read.register(dict)
    def read_content(self, *args, index, **kwargs): return self.content(index)

    @write.register(pd.DataFrame)
    def write_dataframe(self, dataframe, *args, **kwargs):
        with self.mutex:
            dataframe = self.execute(dataframe, *args, **kwargs)
            self.table = dataframe

    @write.register(list)
    def write_records(self, records, *args, **kwargs):
        assert all([isinstance(record, dict) for record in records])
        dataframe = pd.DataFrame.from_records(records, columns=self.header)
        self.write(dataframe, *args, **kwargs)

    @write.register(dict)
    def write_content(self, content, *args, **kwargs):
        dataframe = pd.DataFrame.from_records([content], columns=self.header)
        self.write(dataframe, *args, **kwargs)

    @property
    def index(self): return self.table.index.values
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)

    @property
    @abstractmethod
    def header(self): pass



