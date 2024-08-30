# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Tables Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Tables", "Views"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ViewMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        fields = getattr(cls, "__fields__", {}) | kwargs.get("fields", {})
        values = getattr(cls, "__values__", {})
        update = {key: kwargs.get(key, values.get(key, None)) for key in fields.keys()}
        values = values | update
        cls.__values__ = values
        cls.__fields__ = fields

    def __call__(cls, *args, **kwargs):
        fields = {key: field for key, field in cls.__fields__.items()}
        values = {key: value for key, value in cls.__values__.items()}
        update = {key: kwargs.get(key, value) for key, value in values.items()}
        values = values | update
        instance = super(ViewMeta, cls).__call__(*args, fields=fields, values=values, **kwargs)
        return instance


class View(ABC, metaclass=ViewMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, fields={}, values={}, **kwargs):
        assert set(fields.keys()) == set(values.keys())
        self.__border = "=" * values.get("width", 250)
        self.__values = values
        self.__fields = fields

    def __call__(self, table, *args, **kwargs):
        table = self.execute(table, *args, **kwargs)
        assert isinstance(table, (str, type(None)))
        string = "\n".join([self.border, table, self.border]) if table is not None else ""
        return string + "\n" if bool(string) else string

    @abstractmethod
    def execute(self, table, *args, **kwargs): pass

    @property
    def parameters(self): return {field: self.values[key] for key, field in self.fields.items()}
    @property
    def values(self): return self.__values
    @property
    def fields(self): return self.__fields
    @property
    def border(self): return self.__border


class DataframeView(View, fields={"rows": "max_rows", "columns": "max_cols", "width": "line_width", "formats": "formatters", "numbers": "float_format"}):
    def __init_subclass__(cls, *args, order=[], **kwargs):
        super().__init_subclass__(*args, **kwargs)
        assert isinstance(order, list)
        cls.__order__ = order

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__order = self.__class__.__order__

    def execute(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty):
            return
        columns = self.order if bool(self.order) else list(dataframe.columns)
        length = dataframe.columns.nlevels if isinstance(dataframe.columns, pd.MultiIndex) else 1
        columns = list(self.generator(columns, length)) if length > 1 else list(columns)
        parameters = {key: value for key, value in self.parameters.items()}
        formatters = parameters.pop("formatters", {})
        formatters = zip(self.generator(formatters.keys(), length), formatters.values())
        parameters = parameters | {"formatters": ODict(list(formatters))}
        return dataframe[columns].to_string(**parameters, show_dimensions=True)

    @staticmethod
    def generator(columns, length):
        for column in columns:
            column = tuple([column]) if not isinstance(column, tuple) else column
            column = column + tuple([""]) * (length - len(column))
            yield column if len(column) > 1 else column[0]

    @property
    def order(self): return self.__order


class TableMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is TableMeta for base in cls.__bases__]):
            return
        cls.__tabletype__ = kwargs.get("tabletype", getattr(cls, "__tabletype__", None))
        cls.__tableview__ = kwargs.get("tableview", getattr(cls, "__tableview__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__tabletype__ is not None
        assert cls.__tableview__ is not None
        table = cls.__tabletype__()
        view = cls.__tableview__()
        parameters = dict(mutex=multiprocessing.RLock(), view=view)
        instance = super(TableMeta, cls).__call__(table, *args, **parameters, **kwargs)
        return instance


class Table(ABC, metaclass=TableMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __bool__(self): return not self.empty if self.table is not None else False
    def __str__(self): return self.string
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, table, *args, mutex, view, axes, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__mutex = mutex
        self.__table = table
        self.__view = view
        self.__axes = axes

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
    def mutex(self): return self.__mutex
    @property
    def view(self): return self.__view
    @property
    def axes(self): return self.__axes
    @property
    def name(self): return self.__name


class DataframeMask(ntuple("Mask", "positive negative")):
    def __new__(cls, dataframe, *args, function, **kwargs):
        assert callable(function)
        mask = function(dataframe)
        positive = dataframe.where(mask).dropna(how="all", inplace=False)
        negative = dataframe.where(~mask).dropna(how="all", inplace=False)
        return super().__new__(cls, positive, negative)


class DataframeMerge(ntuple("Merge", "left right")):
    def __new__(cls, left, right, *args, how, on, **kwargs):
        assert isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame) and isinstance(on, list)
        length = lambda cols: int(cols.nlevels) if isinstance(cols, pd.MultiIndex) else 0
        assert length(left.columns) == length(right.columns)
        merge = lambda suffix: lambda col: f"{str(col)}|{suffix}" if col not in on else str(col)
        unmerge = lambda suffix: lambda col: str(col).rstrip(f"|{suffix}") if col not in on else str(col)
        columns = set(left.columns) | set(right.columns)
        left = left.rename(columns=merge("left"), inplace=False, level=0)
        right = right.rename(columns=merge("right"), inplace=False, level=0)
        dataframe = pd.merge(left, right, how=how, on=on)
        left = dataframe.rename(columns=unmerge("left"), inplace=False, level=0)
        right = dataframe.rename(columns=unmerge("right"), inplace=False, level=0)
        left = left[[column for column in columns if column in left.columns]]
        right = right[[column for column in columns if column in right.columns]]
        return super().__new__(cls, left, right)


class DataframeTable(Table, tabletype=pd.DataFrame):
#    def __init__(self, *args, contents=None, **kwargs):
#        super().__init__(*args, **kwargs)
#        assert isinstance(contents, (pd.DataFrame, type(None)))
#        assert contents.columns == self.axes if contents is not None else True
#        if contents is None: self.table.columns = self.axes
#        else: self.table = contents

#    def stack(self, column):
#        if not isinstance(self.table.columns, pd.MultiIndex): return column
#        column = tuple([column]) if not isinstance(column, tuple) else column
#        length = self.columns.nlevels - len(column)
#        column = column + tuple([""]) * length
#        return column

#    def combine(self, other):
#        assert isinstance(other, type(self))
#        assert self.columns == other.columns
#        with self.mutex:
#            dataframes = [self.dataframe, other.dataframe]
#            self.table = pd.concat(dataframes, axis=0)

#    def where(self, function):
#        if not bool(self): return
#        with self.mutex:
#            pass
#            self.table = self.separate(function).positive

#    def remove(self, function):
#        if not bool(self): return
#        with self.mutex:
#            pass
#            self.table = self.separate(function).negative

#    def change(self, function, column, value):
#        if not bool(self): return
#        assert callable(function)
#        column = self.stack(column)
#        with self.mutex:
#            mask = function(self.dataframe)
#            self.table.loc[mask, column] = value

#    def reset(self):
#        if not bool(self): return
#        with self.mutex:
#            self.table.reset_index(drop=True, inplace=True)

#    def unique(self, columns):
#        if not bool(self): return
#        assert isinstance(columns, list)
#        columns = list(map(self.stack, columns))
#        with self.mutex:
#            self.table.drop_duplicates(columns, keep="last", inplace=True)

#    def sort(self, column, reverse):
#        if not bool(self): return
#        assert not isinstance(column, list) and isinstance(reverse, bool)
#        column = self.stack(column)
#        ascending = not bool(reverse)
#        with self.mutex:
#            parameters = dict(ascending=ascending, inplace=True, ignore_index=False)
#            self.table.sort_values(column, axis=0, **parameters)

    @property
    def string(self): return str(self.view(self.table))
    @property
    def empty(self): return bool(self.table.empty)
    @property
    def size(self): return len(self.table.index)

    @property
    def dataframe(self): return self.table
    @property
    def index(self): return self.table.index
    @property
    def columns(self): return self.table.columns


class Tables(object):
    Dataframe = DataframeTable


class Views(object):
    Dataframe = DataframeView



