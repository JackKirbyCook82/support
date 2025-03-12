# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import types
import multiprocessing
import pandas as pd
from itertools import product
from abc import ABC, abstractmethod

from support.mixins import Emptying, Sizing, Partition, Logging, Naming

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Reader", "Routine", "Writer", "Table", "Renderer", "Header", "Stacking", "Layout"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Stacking(Naming, fields=["axis", "columns", "layers"]):
    def __iter__(self): return product(self.columns, self.layers)
    def __contains__(self, column): return column in self.columns
    def __getitem__(self, column):
        if column not in self: return product([column], [""])
        else: return product([column], list(self.layers))

class Header(Naming, fields=["index", "columns", "stacking"]):
    def __len__(self): return len(self.index) + len(self.columns)
    def __iter__(self):
        generator = iter(self.index + self.columns)
        if not bool(self.stacking): yield from generator
        else: yield from iter([value for column in generator for value in self.stacking[column]])


class Layout(Naming, fields=["width", "columns", "rows"]): pass
class Renderer(Naming, fields=["formatting", "layout", "order"]):
    def __new__(cls, *args, order=[], stacking=None, **kwargs):
        split = lambda contents: iter(str(contents).split(" ")) if isinstance(contents, str) else iter(contents)
        formatting = {key: value for keys, value in kwargs.get("formatting", {}).items() for key in split(keys)}
        layout = kwargs.get("layout", {"width": 250, "columns": 35, "rows": 35})
        layout = Layout(*[layout[field] for field in Layout.fields])
        if bool(stacking):
            formatting = {column: function for key, function in formatting.items() for column in stacking[key]}
            order = [column for key in order for column in stacking[key]]
        return super().__new__(cls, formatting=formatting, layout=layout, order=order)

    def __call__(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        numbers = {"float_format": lambda value: f"{value:.02f}"}
        layout = {"line_width": self.layout.width, "max_cols": self.layout.columns, "max_rows": self.layout.rows}
        boundary = str("=") * int(self.layout.width)
        formatting = {"formatters": self.formatting}
        parameters = formatting | numbers | layout
        string = dataframe[self.order].to_string(**parameters, show_dimensions=True)
        strings = [boundary, string, boundary] if bool(string) else []
        string = ("\n".join(strings) + "\n") if bool(strings) else ""
        return string


class Table(ABC):
    def __init__(self, *args, renderer, header, **kwargs):
        self.__data = pd.DataFrame(columns=list(header))
        self.__mutex = multiprocessing.RLock()
        self.__renderer = renderer
        self.__header = header

    def __str__(self): return self.renderer(self.dataframe) if not self.empty else ""
    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __len__(self): return int(self.size) if bool(self) else 0
    def __bool__(self): return not bool(self.empty)

    def __setitem__(self, locator, value): self.set(locator, value)
    def __getitem__(self, locator): return self.get(locator)

    def append(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            columns = list(self.header)
            dataframe = dataframe[columns]
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
        if not isinstance(locator, tuple): index, columns = slice(None, None, None), locator
        elif locator in self.columns: index, columns = slice(None, None, None), locator
        elif len(locator) == 2: index, columns = list(locator)
        else: raise ValueError(locator)
        locate = lambda column: list(self.columns).index(column)
        if not isinstance(columns, list): return index, locate(self.reconcile(columns))
        else: return index, [locate(self.reconcile(column)) for column in columns]

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

    def image(self, mask):
        if not bool(self): return
        assert isinstance(mask, pd.Series)
        with self.mutex:
            dataframe = self.dataframe.where(mask, inplace=False)
            dataframe = dataframe.dropna(how="all", inplace=False)
            return dataframe

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
            column = self.reconcile(column)
            self.dataframe.loc[mask, column] = value

    def unique(self, columns, reverse=True):
        if not bool(self): return
        with self.mutex:
            columns = [self.reconcile(column) for column in columns]
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
            columns = [self.reconcile(column) for column in columns]
            ascending = [not bool(value) for value in reverse]
            parameters = dict(ascending=ascending, axis=0, ignore_index=False, inplace=True)
            self.dataframe.sort_values(columns, **parameters)

    def reindex(self):
        if not bool(self): return
        with self.mutex:
            self.dataframe.reset_index(drop=True, inplace=True)

    def reconcile(self, column):
        if not self.stacked: return column
        column = tuple([column]) if not isinstance(column, tuple) else tuple(column)
        return column + tuple([""]) * (int(self.levels) - len(column))

    @property
    def levels(self): return int(self.columns.nlevels) if isinstance(self.columns, pd.MultiIndex) else 0
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
    def renderer(self): return self.__renderer
    @property
    def header(self): return self.__header
    @property
    def mutex(self): return self.__mutex
    @property
    def data(self): return self.__data
    @property
    def dataframe(self): return self.data

    @data.setter
    def data(self, data): self.__data = data
    @dataframe.setter
    def dataframe(self, dataframe): self.data = dataframe


class Process(Sizing, Emptying, Logging, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.__table = table

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def query(self): return type(self).__query__
    @property
    def table(self): return self.__table


class Routine(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            self.routine(*args, **kwargs)

    @abstractmethod
    def routine(self, *args, **kwargs): pass


class Reader(Process, Partition, ABC, title="Read"):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            dataframes = self.read(*args, **kwargs)
            assert isinstance(dataframes, (pd.DataFrame, types.NoneType))
        if self.empty(dataframes): return
        for query, dataframe in self.partition(dataframes, by=self.query):
            size = self.size(dataframe)
            self.console(f"{str(query)}[{size:.0f}]")
            if self.empty(dataframe): continue
            yield dataframe

    @abstractmethod
    def read(self, *args, **kwargs): pass


class Writer(Process, Partition, ABC, title="Wrote"):
    def execute(self, dataframes, *args, **kwargs):
        assert isinstance(dataframes, (pd.DataFrame, types.NoneType))
        if self.empty(dataframes): return
        for query, dataframe in self.partition(dataframes, by=self.query):
            with self.table.mutex:
                self.write(dataframe, *args, **kwargs)
            size = self.size(dataframe)
            self.console(f"{str(query)}[{size:.0f}]")

    @abstractmethod
    def write(self, content, *args, **kwargs): pass





