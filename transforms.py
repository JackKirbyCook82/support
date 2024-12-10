# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Transform Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from support.mixins import Logging, Sizing, Emptying, Separating

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Pivot", "Unpivot"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Transform(Logging, Sizing, Emptying, Separating, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, header, **kwargs):
        super().__init__(*args, **kwargs)
        self.index, self.columns = header

    def execute(self, dataframes, *args, **kwargs):
        assert isinstance(dataframes, pd.DataFrame)
        if self.empty(dataframes): return
        for group, dataframe in self.separate(dataframes, *args, keys=list(self.query), **kwargs):
            query = self.query(group)
            prior = self.size(dataframe)
            dataframe = self.calculate(dataframe, *args, **kwargs)
            dataframe = dataframe.reset_index(drop=True, inplace=False)
            post = self.size(dataframe)
            string = f"Transformed: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]"
            self.logger.info(string)
            if self.empty(dataframe): continue
            yield dataframe

    @abstractmethod
    def calculate(self, dataframe, *args,  **kwargs): pass


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

