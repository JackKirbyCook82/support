# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Filter Objects
@author: Jack Kirby Cook

"""

from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Sizing, Emptying, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Criterion"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class CriterionMeta(ABCMeta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(CriterionMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        fields = getattr(cls, "__fields__", []) + kwargs.get("fields", [])
        cls.__fields__ = fields

    def __call__(cls, *args, **kwargs):
        criteria = {field: kwargs[field] for field in cls.fields}
        instance = super(CriterionMeta, cls).__call__(criteria, *args, **kwargs)
        return instance

    @property
    def fields(cls): return cls.__fields__


class Criterion(ABC, metaclass=CriterionMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, criteria, *args, **kwargs):
        assert isinstance(criteria, dict)
        self.__criteria = criteria

    def __getitem__(self, key): return self.criteria[key]
    def __call__(self, content, *args, **kwargs):
        mask = self.execute(content)
        content = content.where(mask, axis=0)
        content = content.dropna(how="all", inplace=False)
        return content

    @abstractmethod
    def execute(self, content): pass
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
            results = self.calculate(content, *args, **kwargs)
            post = self.size(results)
            string = f"{str(query)}[{prior:.0f}|{post:.0f}]"
            self.console(string)
            if self.empty(results): continue
            yield results

    def calculate(self, dataframe, *args, **kwargs):
        dataframe = self.criterion(dataframe, *args, **kwargs)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @property
    def criterion(self): return self.__criterion
    @property
    def query(self): return type(self).__query__


