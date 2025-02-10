# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Filter Objects
@author: Jack Kirby Cook

"""

import types
from functools import reduce
from collections import namedtuple as ntuple

from support.mixins import Sizing, Emptying, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Criterion"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Criteria(ntuple("Criteria", "key value function")):
    def __call__(self, content): return self.function(content, self.value)


class CriterionMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = lambda value: isinstance(value, types.LambdaType) and value.__name__ == "<lambda>"
        exclude = [key for key, value in attrs.items() if exclude(value)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(CriterionMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(CriterionMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = {key: function for key, function in getattr(cls, "__registry__", {}).items()}
        function = lambda value: isinstance(value, types.LambdaType) and value.__name__ == "<lambda>"
        updated = {key: value for key, value in attrs.items() if function(value)}
        cls.__functions__ = existing | updated

    def __call__(cls, *args, **kwargs):
        criteria = {Criteria(key, kwargs[key], function) for key, function in cls.functions.items()}
        instance = super(CriterionMeta, cls).__call__(*args, criteria=criteria, **kwargs)
        return instance

    @property
    def functions(cls): return cls.__functions__


class Criterion(object, metaclass=CriterionMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, criteria, **kwargs):
        self.__criteria = criteria

    def __call__(self, content, *args, **kwargs):
        criterion = [criteria(content) for criteria in self.criteria]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        if bool(mask is None): return content
        else: return content.where(mask, axis=0).dropna(how="all", inplace=False)

    @property
    def criteria(self): return self.__criteria


class Filter(Sizing, Emptying, Partition, Logging, title="Filtered"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, criterion, **kwargs):
        assert isinstance(criterion, Criterion)
        super().__init__(*args, **kwargs)
        self.__criterion = criterion

    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for query, content in self.partition(contents, by=self.query):
            prior = self.size(content)
            content = self.calculate(content, *args, **kwargs)
            post = self.size(content)
            string = f"{str(query)}[{prior:.0f}|{post:.0f}]"
            self.console(string)
            if self.empty(content): continue
            yield content

    def calculate(self, dataframe, *args, **kwargs):
        dataframe = self.criterion(dataframe, *args, **kwargs)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @property
    def query(self): return type(self).__query__
    @property
    def criterion(self): return self.__criterion



