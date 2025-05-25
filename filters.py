# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Filter Objects
@author: Jack Kirby Cook

"""

from support.mixins import Sizing, Emptying, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Filter(Sizing, Emptying, Partition, Logging, title="Filtered"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, criteria, **kwargs):
        assert callable(criteria)
        super().__init__(*args, **kwargs)
        self.__criteria = criteria

    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        prior = self.size(contents)
        results = self.calculate(contents, *args, **kwargs)
        post = self.size(results)
        querys = self.keys(contents, by=self.query)
        querys = ",".join(list(map(str, querys)))
        self.console(f"{str(querys)}[{prior:.0f}|{post:.0f}]")
        if self.empty(results): return
        yield results

    def calculate(self, dataframe, *args, **kwargs):
        mask = self.criteria(dataframe)
        dataframe = dataframe.where(mask, axis=0)
        dataframe = dataframe.dropna(how="all", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @property
    def criteria(self): return self.__criteria
    @property
    def query(self): return type(self).__query__


