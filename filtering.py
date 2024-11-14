# -*- coding: utf-8 -*-
"""
Created on Weds Sept 18 2024
@name:   Filtering Objects
@author: Jack Kirby Cook

"""

from functools import reduce
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher
from support.mixins import Logging, Sizing, Emptying

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Criterion"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Criteria(ntuple("Criteria", "variable threshold"), ABC):
    def __call__(self, content): return self.execute(content)

    @abstractmethod
    def execute(self, content): pass

class Floor(Criteria):
    def execute(self, content): return content[self.variable] >= self.threshold

class Ceiling(Criteria):
    def execute(self, content): return content[self.variable] <= self.threshold

class Null(Criteria):
    def execute(self, content): return content[self.variable].notna()

class Criterion(object):
    FLOOR = Floor
    CEILING = Ceiling
    NULL = Null


class Filter(Logging, Sizing, Emptying):
    def __init__(self, *args, criterion={}, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(criterion, dict)
        assert all([issubclass(criteria, Criteria) for criteria in criterion.keys()])
        assert all([isinstance(parameter, (list, dict)) for parameter in criterion.values()])
        criterion = {criteria: parameters if isinstance(parameters, dict) else dict.fromkeys(parameters) for criteria, parameters in criterion.items()}
        criterion = [criteria(variable, threshold) for criteria, parameters in criterion.items() for variable, threshold in parameters.items()]
        self.__criterion = list(criterion)

    def execute(self, query, content, *args, **kwargs):
        if self.empty(content): return
        prior = self.size(content)
        content = self.filter(content)
        content = content.reset_index(drop=True, inplace=False)
        post = self.size(content)
        string = f"Filtered: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]"
        self.logger.info(string)
        if self.empty(content): return
        return content

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

    @property
    def criterion(self): return self.__criterion



