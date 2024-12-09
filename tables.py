# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Logging, Emptying, Sizing, Sourcing
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Reader", "Routine", "Writer", "Process", "Table"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class TableMeta(RegistryMeta, ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is TableMeta for base in bases]):
            return super(TableMeta, mcs).__new__(mcs, name, bases, attrs)
        datatype = kwargs.get("datatype", None)
        if datatype is not None: bases = tuple([Table[datatype]] + list(bases))
        cls = super(TableMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, **kwargs):
        super(TableMeta, cls).__init__(*args, **kwargs)
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))

    def __call__(cls, *args, **kwargs):
        parameters = dict(mutex=multiprocessing.RLock())
        instance = super(TableMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def datatype(cls): return cls.__datatype__


class Table(Logging, ABC, metaclass=TableMeta):
    def __init__(self, *args, data, header, layout, mutex, **kwargs):
        super().__init__(*args, **kwargs)
        self.__header = header
        self.__layout = layout
        self.__mutex = mutex
        self.__data = data

    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __len__(self): return int(self.size) if bool(self) else 0
    def __bool__(self): return not bool(self.empty)
    def __str__(self): return str(self.view)

    def __setitem__(self, locator, value): self.set(locator, value)
    def __getitem__(self, locator): return self.get(locator)

    @abstractmethod
    def set(self, locator, value): pass
    @abstractmethod
    def get(self, locator): pass

    @property
    def view(self):
        boundary = "=" * int(self.layout.width)
        string = self.string if not self.empty else ""
        assert isinstance(string, str)
        strings = [boundary, string, boundary] if bool(string) else []
        string = "\n".join(strings) + "\n" if bool(strings) else ""
        return string

    @property
    def layout(self): return self.__layout
    @property
    def header(self): return self.__header
    @property
    def mutex(self): return self.__mutex
    @property
    def data(self): return self.__data
    @data.setter
    def data(self, data): self.__data = data

    @property
    @abstractmethod
    def string(self): pass
    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass


class TableDataFrame(Table, register=pd.DataFrame):
    def __init__(self, *args, header, layout, **kwargs):
        assert all([hasattr(layout, attribute) for attribute in ("order", "numbers", "width", "columns", "rows")])
        data = pd.DataFrame(columns=list(header))
        parameters = dict(data=data, header=header, layout=layout)
        super().__init__(*args, **parameters, **kwargs)

    def append(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframe = dataframe[list(self.header)]
            if bool(self.dataframe.empty): self.dataframe = dataframe
            else: self.dataframe = pd.concat([self.dataframe, dataframe], axis=0, ignore_index=True)

    def get(self, locator):
        with self.mutex:
            index, columns = self.locate(locator)
            return self.dataframe.iloc[index, columns]

    def set(self, locator, value):
        with self.mutex:
            index, columns = self.locate(locator)
            self.dataframe.iloc[index, columns] = value

    def locate(self, locator):
        stack = lambda column: self.stack(column)
        locate = lambda column: list(self.columns).index(stack(column))
        if not isinstance(locator, tuple): index, columns = slice(None, None, None), locator
        elif locator in self.columns: index, columns = slice(None, None, None), locator
        elif len(locator) == 2: index, columns = list(locator)
        else: raise ValueError(locator)
        if not isinstance(columns, list): return index, locate(columns)
        else: return index, [locate(column) for column in columns]

    def retain(self, mask):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            self.dataframe.where(mask, inplace=True)
            self.dataframe.dropna(how="all", inplace=True)

    def discard(self, mask):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            self.dataframe.where(~mask, inplace=True)
            self.dataframe.dropna(how="all", inplace=True)

    def take(self, mask):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            dataframe = self.dataframe.where(mask, inplace=False)
            dataframe = dataframe.dropna(how="all", inplace=False)
            self.dataframe.where(~mask, inplace=True)
            self.dataframe.dropna(how="all", inplace=True)
            return dataframe

    def modify(self, mask, column, value):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            column = self.stack(column)
            self.dataframe.loc[mask, column] = value

    def unique(self, columns, reverse=True):
        if not bool(self): return
        with self.mutex:
            columns = [self.stack(column) for column in columns]
            keep = ("last" if bool(reverse) else "first")
            parameters = dict(keep=keep, inplace=True)
            self.dataframe.drop_duplicates(columns, **parameters)

    def sort(self, columns, reverse):
        if not bool(self): return
        assert isinstance(columns, (list, str)) and isinstance(reverse, (list, bool))
        columns = [columns] if isinstance(columns, str) else columns
        reverse = [reverse] if isinstance(reverse, bool) else reverse
        reverse = reverse * len(columns) if len(reverse) == 1 else reverse
        assert len(columns) == len(reverse)
        with self.mutex:
            columns = [self.stack(column) for column in columns]
            ascending = [not bool(value) for value in reverse]
            parameters = dict(ascending=ascending, axis=0, ignore_index=False, inplace=True)
            self.dataframe.sort_values(columns, **parameters)

    def reindex(self):
        if not bool(self): return
        with self.mutex:
            self.dataframe.reset_index(drop=True, inplace=True)

    def stack(self, column):
        if not bool(self.stacked): return column
        column = tuple([column]) if not isinstance(column, tuple) else column
        return column + tuple([""]) * (int(self.stacking) - len(column))

    @property
    def string(self):
        order = [self.stack(column) for column in self.layout.order]
        formats = {"formatters": {self.stack(column): value for column, value in self.layout.formats.items()}}
        numbers = {"float_format": self.layout.numbers}
        width = {"line_width": self.layout.width}
        columns = {"max_cols": self.layout.columns}
        rows = {"max_rows": self.layout.rows}
        parameters = formats | numbers | width | columns | rows
        string = self.dataframe[order].to_string(**parameters, show_dimensions=True)
        return string

    @property
    def stacking(self): return int(self.columns.nlevels) if bool(self.stacked) else 0
    @property
    def stacked(self): return isinstance(self.columns, pd.MultiIndex)
    @property
    def empty(self): return bool(self.data.empty)
    @property
    def size(self): return len(self.data.index)
    @property
    def columns(self): return self.data.columns
    @property
    def index(self): return self.data.index

    @property
    def dataframe(self): return self.data
    @dataframe.setter
    def dataframe(self, dataframe): self.data = dataframe


class Process(Logging, Sizing, Emptying, Sourcing, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        cls.title = kwargs.get("title", getattr(cls, "title", None))
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.__table = table

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def table(self): return self.__table


class Reader(Process, ABC, title="Read"):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            contents = self.read(*args, **kwargs)
            if self.empty(contents): return
            for query, content in self.source(contents, *args, query=self.query, **kwargs):
                size = self.size(content)
                string = f"{str(self.title)}: {repr(self)}|{str(query)}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(content): continue
                yield content

    @abstractmethod
    def read(self, *args, **kwargs): pass


class Routine(Process, ABC, title="Performed"):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            self.routine(*args, **kwargs)

    @abstractmethod
    def routine(self, *args, **kwargs): pass


class Writer(Process, ABC, title="Wrote"):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        with self.table.mutex:
            for query, content in self.source(contents, *args, query=self.query, **kwargs):
                self.write(content, *args, **kwargs)
                size = self.size(content)
                string = f"{str(self.title)}: {repr(self)}|{str(query)}[{size:.0f}]"
                self.logger.info(string)

    @abstractmethod
    def write(self, content, *args, **kwargs): pass





