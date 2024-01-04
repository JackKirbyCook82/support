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
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __init__(self, *args, capacity=None, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        name = str(self.name).replace("Table", "Lock")
        self.__mutex = Lock(name=name, timeout=timeout)
        self.__table = pd.DataFrame()
        self.__capacity = capacity

    def read(self, *args, **kwargs):
        with self.mutex:
            return self.table

    def write(self, table, *args, **kwargs):
        with self.mutex:
            table = self.combine(table, *args, **kwargs)
            table = self.parser(table, *args, **kwargs)
            table = self.format(table, *args, **kwargs)
            self.table = table

    @abstractmethod
    def combine(self, table, *args, **kwargs): pass

    @staticmethod
    def parser(table, *args, **kwargs): return table
    @staticmethod
    def format(table, *args, **kwargs): return table

    @property
    def capacity(self): return self.__capacity
    @property
    def mutex(self): return self.__mutex
    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table


class DataframeTable(Table):
    def combine(self, table): return pd.concat([self.table, table], axis=0)

    @property
    def size(self): return len(self.table.index)
    @property
    def empty(self): return bool(self.table.empty)



