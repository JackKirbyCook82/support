# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Fields
from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Options", "Tables"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class OptionsMeta(ABCMeta): pass
class Options(Fields, metaclass=OptionsMeta): pass
class DataframeOptions(Options, fields=["rows", "columns", "width", "formats", "numbers"]):
    @property
    def parameters(self):
        return dict(max_rows=self.rows, max_cols=self.columns, line_width=self.width, formatters=self.formats, float_format=self.numbers)


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is TableMeta for base in cls.__bases__]):
            return
        cls.__tabletype__ = kwargs.get("tabletype", getattr(cls, "__tabletype__", None))
        cls.__options__ = kwargs.get("options", getattr(cls, "__options__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__tabletype__ is not None
        assert cls.__options__ is not None
        instance = cls.__tabletype__()
        parameters = dict(mutex=multiprocessing.RLock(), options=cls.__options__)
        instance = super(TableMeta, cls).__call__(instance, *args, **parameters, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __bool__(self): return not self.empty if self.table is not None else False
    def __str__(self): return self.string
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, instance, *args, mutex, options, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__options = options
        self.__table = instance
        self.__mutex = mutex

    @property
    @abstractmethod
    def string(self): pass
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
    def options(self): return self.__options
    @property
    def mutex(self): return self.__mutex
    @property
    def name(self): return self.__name


class DataframeTable(Table, tabletype=pd.DataFrame):
    def remove(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframe = self.table.drop(dataframe.index, inplace=False)
            self.table = dataframe

    def where(self, mask):
        assert isinstance(mask, pd.Series)
        with self.mutex:
            dataframe = self.table.where(mask)
            dataframe = dataframe.dropna(how="all", inplace=False)
            return dataframe

    def concat(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframe = pd.concat([self.table, dataframe], axis=0)
            self.table = dataframe

    def update(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframe = pd.concat([self.table, dataframe], axis=0)
            dataframe = dataframe[~dataframe.index.duplicated(keep="last")]
            self.table = dataframe

    def sort(self, column, reverse=False):
        assert column in self.table.columns
        with self.mutex:
            self.table.sort_values(column, axis=0, ascending=not bool(reverse), inplace=True, ignore_index=False)

    @property
    def string(self):
        dataframe = self.table.reset_index(drop=False, inplace=False)
        string = dataframe.to_string(**self.options.parameters, show_dimensions=True)
        return string

    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)


class Tables(object):
    Dataframe = DataframeTable


class Options(object):
    Dataframe = DataframeOptions



