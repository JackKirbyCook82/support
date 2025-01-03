# -*- coding: utf-8 -*-
"""
Created on Tues Dec 10 2024
@name:   Filter Objects
@author: Jack Kirby Cook

"""

from functools import reduce

from support.mixins import Logging, Sizing, Emptying, Separating

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Filter(Logging, Sizing, Emptying, Separating):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, criterion, **kwargs):
        assert isinstance(criterion, list) or callable(criterion)
        assert all([callable(function) for function in criterion]) if isinstance(criterion, list) else callable(criterion)
        super().__init__(*args, **kwargs)
        self.__criterion = list(criterion) if isinstance(criterion, list) else [criterion]

    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for group, content in self.separate(contents, *args, fields=self.fields, **kwargs):
            query = self.query(group)
            prior = self.size(content)
            content = self.calculate(content, *args, **kwargs)
            content = content.reset_index(drop=True, inplace=False)
            post = self.size(content)
            string = f"Filtered: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]"
            self.logger.info(string)
            if self.empty(content): continue
            yield content

    def calculate(self, content, *args, **kwargs):
        criterion = [criteria(content) for criteria in self.criterion]
        mask = reduce(lambda x, y: x & y, criterion) if bool(criterion) else None
        if bool(mask is None): return content
        else: return content.where(mask, axis=0).dropna(how="all", inplace=False)

    @property
    def fields(self): return list(type(self).__query__)
    @property
    def query(self): return type(self).__query__
    @property
    def criterion(self): return self.__criterion

