# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Transform Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from support.mixins import Sizing, Emptying, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Pivot", "Unpivot"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Transform(Sizing, Emptying, Partition, Logging, ABC, title="Transformed"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, header, **kwargs):
        super().__init__(*args, **kwargs)
        index, columns = list(header)
        self.__columns = columns
        self.__index = index

    def execute(self, dataframes, *args, **kwargs):
        assert isinstance(dataframes, pd.DataFrame)
        if self.empty(dataframes): return
        for query, dataframe in self.partition(dataframes, by=self.query):
            prior = self.size(dataframe)
            dataframe = self.calculate(dataframe, *args, **kwargs)
            dataframe = dataframe.reset_index(drop=True, inplace=False)
            post = self.size(dataframe)
            string = f"{str(query)}[{prior:.0f}|{post:.0f}]"
            self.console(string)
            if self.empty(dataframe): continue
            yield dataframe

    @abstractmethod
    def calculate(self, dataframe, *args,  **kwargs): pass

    @property
    def query(self): return type(self).__query__
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class Pivot(Transform, title="Pivoted"):
    def calculate(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({self.index} | set(self.columns))
        dataframe = dataframe.pivot(index=list(index), columns=[self.index])
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class Unpivot(Transform, title="Unpivoted"):
    def calculate(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        dataframe.index.name = "index"
        level = list(dataframe.columns.names).index(self.index)
        index = set(dataframe.columns) - set(self.columns)
        columns = set([values for values in dataframe.columns.values if bool(values[level])])
        index = dataframe[list(index)].stack().reset_index(drop=False, inplace=False).drop(columns=self.index)
        columns = dataframe[list(columns)].stack().reset_index(drop=False, inplace=False)
        dataframe = pd.merge(index, columns, how="outer", on="index").drop(columns="index")
        return dataframe



