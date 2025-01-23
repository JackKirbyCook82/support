# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Transform Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd
from abc import ABC, abstractmethod

from support.mixins import Sizing, Emptying, Partition

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Pivot", "Unpivot"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Transform(Partition, Sizing, Emptying, ABC):
    def __init__(self, *args, header, **kwargs):
        super().__init__(*args, **kwargs)
        index, columns = list(header)
        self.__columns = columns
        self.__index = index

    def execute(self, dataframes, *args, **kwargs):
        assert isinstance(dataframes, pd.DataFrame)
        if self.empty(dataframes): return
        for query, dataframe in self.separate(dataframes, *args, **kwargs):
            prior = self.size(dataframe)
            dataframe = self.calculate(dataframe, *args, **kwargs)
            dataframe = dataframe.reset_index(drop=True, inplace=False)
            post = self.size(dataframe)
            string = f"Transformed: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]"
            __logger__.info(string)
            if self.empty(dataframe): continue
            yield dataframe

    @abstractmethod
    def calculate(self, dataframe, *args,  **kwargs): pass

    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class Pivot(Transform):
    def calculate(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({self.index} | set(self.columns))
        dataframe = dataframe.pivot(index=list(index), columns=[self.index])
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class Unpivot(Transform):
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

