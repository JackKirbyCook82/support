# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, abstractmethod

from support.mixins import Logging, Emptying, Sizing, Separating

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Reader", "Routine", "Writer", "Table"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Table(Logging, ABC):
    def __init__(self, *args, header, layout, **kwargs):
        assert all([hasattr(layout, attribute) for attribute in ("order", "numbers", "width", "columns", "rows")])
        super().__init__(*args, **kwargs)
        self.__mutex = multiprocessing.RLock()
        self.__data = pd.DataFrame(columns=list(header))
        self.__header = header
        self.__layout = layout

    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __len__(self): return int(self.size) if bool(self) else 0
    def __bool__(self): return not bool(self.empty)
    def __str__(self): return str(self.view)

    def __setitem__(self, locator, value): self.set(locator, value)
    def __getitem__(self, locator): return self.get(locator)

    def draw(self, *args, **kwargs):
        if self.empty: return ""
        order = [self.stack(column) for column in self.layout.order]
        formats = {self.stack(column): value for column, value in self.layout.formats.items()}
        formats = {"formatters": kwargs.get("formats", formats)}
        numbers = {"float_format": kwargs.get("numbers", self.layout.numbers)}
        width = {"line_width": kwargs.get("width", self.layout.width)}
        columns = {"max_cols": kwargs.get("columns", self.layout.columns)}
        rows = {"max_rows": kwargs.get("rows", self.layout.rows)}
        parameters = formats | numbers | width | columns | rows
        string = self.dataframe[order].to_string(**parameters, show_dimensions=True)
        boundary = "=" * int(self.layout.width)
        strings = [boundary, string, boundary] if bool(string) else []
        string = "\n".join(strings) + "\n" if bool(strings) else ""
        return string

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


class Process(Logging, Sizing, Emptying, Separating, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, table, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.table = table

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Reader(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutux: contents = self.read(*args, **kwargs)
        if self.empty(contents): return
        for group, content in self.source(contents, *args, keys=list(self.query), **kwargs):
            query = self.query(group)
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
        with self.table.mutex: self.invoke(*args, **kwargs)

    @abstractmethod
    def invoke(self, *args, **kwargs): pass


class Writer(Process, ABC):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for group, content in self.separate(contents, *args, keys=list(self.query), **kwargs):
            query = self.query(group)
            with self.table.mutex: self.write(content, *args, **kwargs)
            size = self.size(content)
            string = f"Wrote: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)

    @abstractmethod
    def write(self, content, *args, **kwargs): pass





