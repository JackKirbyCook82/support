# -*- coding: utf-8 -*-
"""
Created on Weds Sept 18 2024
@name:   Operation Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from functools import reduce
from abc import ABC, abstractmethod

from support.mixins import Logging, Sizing, Emptying, Sourcing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Pivot", "Unpivot"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Operation(Logging, Sizing, Emptying, Sourcing, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.title = kwargs.get("title", getattr(cls, "title", None))
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for query, content in self.source(contents, *args, query=self.query, **kwargs):
            prior = self.size(content)
            content = self.calculate(content, *args, **kwargs)
            content = content.reset_index(drop=True, inplace=False)
            post = self.size(content)
            string = f"{str(self.title)}: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]"
            self.logger.info(string)
            if self.empty(content): continue
            yield content

    @abstractmethod
    def calculate(self, content, *args, **kwargs): pass


class Filter(Operation, title="Filtered"):
    def __init__(self, *args, criterion=[], **kwargs):
        assert isinstance(criterion, list) or callable(criterion)
        assert all([callable(function) for function in criterion]) if isinstance(criterion, list) else True
        super().__init__(*args, **kwargs)
        self.criterion = list(criterion) if isinstance(criterion, list) else [criterion]

    def calculate(self, dataframe, *args, **kwargs):
        criterion = [criteria(dataframe) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        if bool(mask is None): return dataframe
        else: return dataframe.where(mask, axis=0).dropna(how="all", inplace=False)


class Pivot(Operation, title="Pivoted"):
    def __init__(self, *args, pivot, **kwargs):
        super().__init__(*args, **kwargs)
        index, columns = pivot
        self.columns = columns
        self.index = index

    def calculate(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({self.index} | set(self.columns))
        dataframe = dataframe.pivot(index=list(index), columns=[self.index])
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class Unpivot(Operation, title="UnPivoted"):
    def __init__(self, *args, unpivot, **kwargs):
        super().__init__(*args, **kwargs)
        index, columns = unpivot
        self.columns = columns
        self.index = index

    def calculate(self, dataframe, *args, **kwargs):
        dataframe.index.name = "index"
        level = list(dataframe.columns.names).index(self.index)
        index = set(dataframe.columns) - set(self.columns)
        columns = set([values for values in dataframe.columns.values if bool(values[level])])
        index = dataframe[list(index)].stack().reset_index(drop=False, inplace=False).drop(columns=self.index)
        columns = dataframe[list(columns)].stack().reset_index(drop=False, inplace=False)
        dataframe = pd.merge(index, columns, how="outer", on="index").drop(columns="index")
        return dataframe



