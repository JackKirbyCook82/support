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
from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Emptying, Sizing, Partition, Logging, Naming

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Reader", "Routine", "Writer", "Table", "Renderer", "Header", "Stacking", "Layout"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Header(Naming, fields=["index", "columns"]):
    def __new__(cls, *args, index=[], columns=[], stacking=None, **kwargs):
        if bool(stacking):
            generator = lambda value: product([value], stacking.layers) if value in stacking.columns else product([value], [""])
            columns = [value for key in columns for value in generator(key)]
            index = [value for key in index for value in generator(key)]
        return super().__new__(cls, index=index, columns=columns)

    def __len__(self): return len(self.index) + len(self.columns)
    def __iter__(self): return iter(self.index + self.columns)


class Layout(Naming, fields=["width", "space", "columns", "rows"]): pass
class Stacking(Naming, fields=["axis", "columns", "layers"]): pass
class Renderer(Naming, fields=["formatters", "layout", "order"]):
    def __new__(cls, *args, layout={}, order=[], stacking=None, **kwargs):
        split = lambda contents: iter(str(contents).split(" ")) if isinstance(contents, str) else iter(contents)
        formatters = {key: value for keys, value in kwargs.get("formatters", {}).items() for key in split(keys)}
        if bool(stacking):
            generator = lambda column: product([column], stacking.layers) if column in stacking.columns else product([column], [""])
            formatters = {column: function for key, function in formatters.items() for column in generator(key)}
            order = [column for key in order for column in generator(key)]
        return super().__new__(cls, formatters=formatters, layout=layout, order=order)

    def __call__(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        numbers = {"float_format": lambda value: f"{value:.02f}"}
        layout = {"line_width": self.layout.width, "col_space": self.layout.space, "max_cols": self.layout.columns, "max_rows": self.layout.rows}
        boundary = str("=") * int(self.layout.width)
        formatters = {"formatters": self.formatters}
        parameters = formatters | numbers | layout
        string = dataframe[self.order].to_string(**parameters, show_dimensions=True)
        strings = [boundary, string, boundary] if bool(string) else []
        string = ("\n".join(strings) + "\n") if bool(strings) else ""
        return string


class TableMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        renderer = Renderer(*args, **kwargs)
        header = Header(*args, **kwargs)
        parameters = dict(renderer=renderer, header=header)
        instance = super(TableMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init__(self, *args, renderer, header, **kwargs):
        self.__data = pd.DataFrame(columns=list(header))
        self.__mutex = multiprocessing.RLock()
        self.__renderer = renderer

    def __str__(self): return self.renderer(self.dataframe) if not self.empty else ""
    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __len__(self): return int(self.size) if bool(self) else 0
    def __bool__(self): return not bool(self.empty)

    def __setitem__(self, column, value): self.set(column, value)
    def __getitem__(self, column): return self.get(column)

    def get(self, column):
        with self.mutex:
            column = self.reconcile(column)
            return self.dataframe[column]

    def set(self, column, value):
        assert isinstance(value, pd.Series) and len(value) == len(self)
        with self.mutex:
            column = self.reconcile(column)
            self.dataframe[column] = value
            self.renderer.order.append(column)

    def append(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        with self.mutex:
            dataframe = dataframe[self.dataframe.columns]
            if bool(self.dataframe.empty): self.dataframe = dataframe
            else: self.dataframe = pd.concat([self.dataframe, dataframe], axis=0, ignore_index=True)

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

    def portray(self, mask):
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
            columns = list(map(self.reconcile, columns))
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
            columns = list(map(self.reconcile, columns))
            ascending = [not bool(value) for value in reverse]
            parameters = dict(ascending=ascending, axis=0, ignore_index=False, inplace=True)
            self.dataframe.sort_values(columns, **parameters)

    def reconcile(self, column):
        if not self.stacked: return column
        column = tuple([column]) if not isinstance(column, tuple) else tuple(column)
        return column + tuple([""]) * (int(self.levels) - len(column))

    def reindex(self):
        if not bool(self): return
        with self.mutex:
            self.dataframe.reset_index(drop=True, inplace=True)

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
    def mutex(self): return self.__mutex
    @property
    def dataframe(self): return self.__data
    @property
    def data(self): return self.__data

    @dataframe.setter
    def dataframe(self, dataframe): self.__data = dataframe
    @data.setter
    def data(self, data): self.__data = data


class Process(Sizing, Emptying, Partition, Logging, ABC):
    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.__table = table

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def table(self): return self.__table


class Routine(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        self.routine(*args, **kwargs)

    @abstractmethod
    def routine(self, *args, **kwargs): pass


class Reader(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.table): return
        dataframe = self.read(*args, **kwargs)
        assert isinstance(dataframe, (pd.DataFrame, types.NoneType))
        if self.empty(dataframe): return
        yield dataframe

    @abstractmethod
    def read(self, *args, **kwargs): pass


class Writer(Process, ABC):
    def execute(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, (pd.DataFrame, types.NoneType))
        if self.empty(dataframe): return
        self.write(dataframe, *args, **kwargs)

    @abstractmethod
    def write(self, dataframe, *args, **kwargs): pass






