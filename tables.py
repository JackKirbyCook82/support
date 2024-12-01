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
from support.dispatchers import typedispatcher
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Reader", "Routine", "Writer", "Process", "Table", "View"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ViewMeta(RegistryMeta, ABCMeta):
    def __init__(cls, *args, **kwargs):
        super(ViewMeta, cls).__init__(*args, **kwargs)
        cls.__layouttype__ = kwargs.get("layouttype", getattr(cls, "__layouttype__", None))

    def __call__(cls, *args, **kwargs):
        layout = cls.layouttype(*args, **kwargs)
        parameters = dict(layout=layout)
        instance = super(TableMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def layouttype(cls): return cls.__layouttype__


class View(Logging, Sizing, Emptying, ABC, metaclass=ViewMeta):
    def __init__(self, *args, layout, **kwargs):
        super().__init__(*args, **kwargs)
        self.__layout = layout

    def __repr__(self): return f"{str(self.name)}"
    def __call__(self, table, *args, **kwargs):
        if self.empty(table): return ""
        table = self.execute(table, *args, **kwargs)
        assert isinstance(table, str)
        string = "\n".join(["=" * 250, table, "=" * 250]) if table is not None else ""
        return string + "\n" if bool(string) else string

    @abstractmethod
    def execute(self, table, *args, **kwargs): pass
    @property
    def layout(self): return self.__layout


class ViewDataframe(View, register=pd.DataFrame):
    def execute(self, dataframe, *args, **kwargs):
        stacking = self.stacking(dataframe)
        order = [self.stack(column, stacking) for column in getattr(self.layout, "order", [])]
        formats = {"formatters": {self.stack(key, stacking): value for key, value in getattr(self.layout, "formats", {}).items()}}
        numbers = {"float_format": getattr(self.layout, "numbers", lambda column: f"{column:.02f}")}
        width = {"line_width": getattr(self.layout, "width", 250)}
        columns = {"max_cols", getattr(self.layout, "columns", 30)}
        rows = {"max_rows": getattr(self.layout, "rows", 30)}
        parameters = formats | numbers | width | columns | rows
        string = dataframe[order].to_string(**parameters, show_dimensions=True)
        return string

    @staticmethod
    def stacking(dataframe): return int(dataframe.columns.nlevels) if isinstance(dataframe.columns, pd.MultiIndex) else 0
    @staticmethod
    def stack(column, stacking):
        if not bool(stacking): return column
        column = tuple([column]) if not isinstance(column, tuple) else column
        return column + tuple([""]) * (int(stacking) - len(column))


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
        cls.__headertype__ = kwargs.get("headertype", getattr(cls, "__headertype__", None))
        cls.__viewtype__ = kwargs.get("viewtype", getattr(cls, "__viewtype__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))

    def __call__(cls, *args, **kwargs):
        header = cls.headertype(*args, **kwargs)
        view = cls.viewtype(*args, **kwargs)
        data = cls.datatype(columns=list(header))
        mutex = multiprocessing.RLock()
        parameters = dict(mutex=mutex, header=header, view=view, data=data)
        instance = super(TableMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def headertype(cls): return cls.__headertype__
    @property
    def viewtype(cls): return cls.__viewtype__
    @property
    def datatype(cls): return cls.__datatype__


class Table(Logging, ABC, metaclass=TableMeta):
    def __init__(self, *args, data, view, header, mutex, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mutex = mutex
        self.__header = header
        self.__view = view
        self.__data = data

    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __len__(self): return int(self.size) if bool(self) else 0
    def __str__(self): return str(self.view(self.data))
    def __bool__(self): return not bool(self.empty)

    def __setitem__(self, locator, value): self.set(locator, value)
    def __getitem__(self, locator): return self.get(locator)

    @abstractmethod
    def set(self, locator, value): pass
    @abstractmethod
    def get(self, locator): pass

    @property
    def mutex(self): return self.__mutex
    @property
    def header(self): return self.__header
    @property
    def view(self): return self.__view
    @property
    def data(self): return self.__data
    @data.setter
    def data(self, data): self.__data = data

    @property
    @abstractmethod
    def empty(self): pass
    @property
    @abstractmethod
    def size(self): pass


class TableDataFrame(Table, register=pd.DataFrame):
    def combine(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            self.dataframe = pd.concat([self.dataframe, dataframe], axis=0) if bool(self) else dataframe

    def get(self, locator):
        index, columns = locator
        assert isinstance(index, slice) and isinstance(columns, (str, list))
        columns = self.locate(columns)
        return self.dataframe.iloc[index, columns]

    def set(self, locator, value):
        index, columns = locator
        assert isinstance(index, slice) and isinstance(columns, (str, list))
        columns = self.locate(columns)
        self.dataframe.iloc[index, columns] = value

    def where(self, mask):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            self.dataframe.where(mask, inplace=True)
            self.dataframe.dropna(how="all", inplace=True)

    def remove(self, mask):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            self.dataframe.where(~mask, inplace=True)
            self.dataframe.dropna(how="all", inplace=True)

    def extract(self, mask):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            dataframe = self.dataframe.where(mask, inplace=False)
            dataframe = dataframe.dropna(how="all", inplace=False)
            self.dataframe.where(~mask, inplace=True)
            self.dataframe.dropna(how="all", inplace=True)
            return dataframe

    def change(self, mask, column, value):
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

    def reset(self):
        if not bool(self): return
        with self.mutex:
            self.dataframe.reset_index(drop=True, inplace=True)

    def clear(self):
        if not bool(self): return
        with self.mutex:
            index = self.index
            self.dataframe.drop(index, inplace=True)

    def stack(self, column):
        if not bool(self.stacked): return column
        column = tuple([column]) if not isinstance(column, tuple) else column
        return column + tuple([""]) * (int(self.stacking) - len(column))

    @typedispatcher
    def locate(self, columns): raise TypeError(type(columns))
    @locate.register(list)
    def __multiple(self, columns): return [self.locate(column) for column in columns]
    @locate.register(str)
    def __single(self, column):
        column = self.stack(column)
        column = list(self.columns).index(column)
        return column

    @property
    def dataframe(self): return self.data
    @dataframe.setter
    def dataframe(self, dataframe): self.data = dataframe

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


class Process(Logging, Sizing, Emptying, Sourcing, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.__table = table

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def table(self): return self.__table


class Reader(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            contents = self.read(*args, **kwargs)
            if self.empty(contents): return
            for query, content in self.source(contents, *args, query=self.query, **kwargs):
                size = self.size(content)
                string = f"Read: {repr(self)}|{str(query)}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(content): continue
                yield content

    @abstractmethod
    def read(self, *args, **kwargs): pass


class Routine(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            self.routine(*args, **kwargs)

    @abstractmethod
    def routine(self, *args, **kwargs): pass


class Writer(Process, ABC):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        with self.table.mutex:
            for query, content in self.source(contents, *args, query=self.query, **kwargs):
                self.write(content, *args, **kwargs)
                size = self.size(content)
                string = f"Wrote: {repr(self)}|{str(query)}[{size:.0f}]"
                self.logger.info(string)

    @abstractmethod
    def write(self, content, *args, **kwargs): pass





