# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Filter Objects
@author: Jack Kirby Cook

"""

from functools import reduce

from support.mixins import Sizing, Emptying, Partition

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Filter(Partition, Sizing, Emptying, title="Filtered"):
    def __init__(self, *args, criterion, **kwargs):
        assert isinstance(criterion, list) or callable(criterion)
        assert all([callable(function) for function in criterion]) if isinstance(criterion, list) else callable(criterion)
        super().__init__(*args, **kwargs)
        self.__criterion = list(criterion) if isinstance(criterion, list) else [criterion]

    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for query, content in self.partition(contents):
            prior = self.size(content)
            content = self.calculate(content, *args, **kwargs)
            content = content.reset_index(drop=True, inplace=False)
            post = self.size(content)
            string = f"{str(query)}[{prior:.0f}|{post:.0f}]"
            self.console(string)
            if self.empty(content): continue
            yield content

    def calculate(self, content, *args, **kwargs):
        criterion = [criteria(content) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        if bool(mask is None): return content
        else: return content.where(mask, axis=0).dropna(how="all", inplace=False)

    @property
    def criterion(self): return self.__criterion



