# -*- coding: utf-8 -*-
"""
Created on Weds Sept 18 2024
@name:   Filtering Objects
@author: Jack Kirby Cook

"""

from functools import reduce

from support.dispatchers import typedispatcher
from support.mixins import Logging, Sizing, Emptying, Sourcing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Filter(Logging, Sizing, Emptying, Sourcing):
    def __init_subclass__(cls, *args, **kwargs):
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, criterion=[], **kwargs):
        assert isinstance(criterion, list) or callable(criterion)
        assert all([callable(function) for function in criterion]) if isinstance(criterion, list) else True
        super().__init__(*args, **kwargs)
        self.criterion = list(criterion) if isinstance(criterion, list) else [criterion]

    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for query, content in self.source(contents, *args, query=self.query, **kwargs):
            prior = self.size(content)
            content = self.filter(content)
            content = content.reset_index(drop=True, inplace=False)
            post = self.size(content)
            string = f"Filtered: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]"
            self.logger.info(string)
            if self.empty(content): continue
            yield content

    def filter(self, content):
        mask = self.mask(content)
        content = self.where(content, mask=mask)
        return content

    def mask(self, content):
        criterion = [criteria(content) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        return mask

    @typedispatcher
    def where(self, dataframe, mask=None):
        if bool(mask is None): return dataframe
        else: return dataframe.where(mask).dropna(how="all", inplace=False)




