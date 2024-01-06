# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""
import numpy as np
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
    def __bool__(self): return not self.empty if self.table is not None else False
    def __len__(self): return self.size

    def __init__(self, contents, *args, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        name = str(self.name).replace("Table", "Lock")
        assert isinstance(contents, self.type)
        self.__mutex = Lock(name=name, timeout=timeout)
        self.__table = contents

    def read(self, *args, **kwargs):
        with self.mutex:
            return self.table

    def write(self, table, *args, **kwargs):
        assert isinstance(table, self.type)
        with self.mutex:
            table = self.execute(table, *args, **kwargs)
            self.table = table

    @abstractmethod
    def execute(self, other, *args, **kwargs): pass

    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table
    @property
    def mutex(self): return self.__mutex


class DataframeTable(Table, ABC, type=pd.DataFrame):
    def __init__(self, *args, **kwargs):
        dataframe = pd.DataFrame(columns=self.header)
        super().__init__(dataframe, *args, **kwargs)

    def execute(self, dataframe, *args, **kwargs):
        start = self.table.index.max() + 1 if not bool(self.table.empty) else 0
        index = np.arange(start, start + len(dataframe.index))
        dataframe = dataframe.set_index(index, drop=True, inplace=False)[self.header]
        dataframe = pd.concat([self.table, dataframe], axis=0)
        return dataframe

    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)

    @property
    @abstractmethod
    def header(self): pass





