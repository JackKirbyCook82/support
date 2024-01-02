# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC

from support.pipelines import Producer, Consumer
from support.locks import Lock

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Table", "TableReader", "TableWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class TableReader(Producer, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(source, Table)
        self.__source = source

#    def read(self, *args, **kwargs): pass
#    def reader(self, *args, **kwargs): pass

    @property
    def source(self): return self.__source
    @property
    def table(self): return self.__source


class TableWriter(Consumer, ABC):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(destination, Table)
        self.__destination = destination

#    def write(self, *args, **kwargs): pass
#    def writer(self, *args, **kwargs): pass

    @property
    def destination(self): return self.__destination
    @property
    def table(self): return self.__destination


class Table(object):
    def __bool__(self): return not self.empty
    def __repr__(self): return self.name
    def __len__(self): return self.size

    def __init__(self, *args, timeout=None, capacity=None, **kwargs):
        table_name = kwargs.get("name", self.__class__.__name__)
        lock_name = str(table_name).replace("Table", "Lock")
        self.__table = pd.DataFrame()
        self.__mutex = Lock(name=lock_name, timeout=timeout)
        self.__capacity = capacity
        self.__name = table_name

#    def done(self): pass
#    def get(self): pass
#    def put(self, content): pass

    @property
    def size(self): return len(self.table.index)
    @property
    def empty(self): return bool(self.table.empty)

    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table

    @property
    def capacity(self): return self.__capacity
    @property
    def mutex(self): return self.__mutex








