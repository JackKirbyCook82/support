# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 2024
@name:   Process Objects
@author: Jack Kirby Cook

"""

import inspect
import logging
from abc import ABC, abstractmethod

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Operation", "Feed"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Process(ABC):
    def __init__(self, feed, other):
        assert isinstance(feed, Feed) and isinstance(other, (list, Operation))
        assert all([isinstance(operation, Operation) for operation in other]) if isinstance(other, list) else True
        operations = [other] if isinstance(other, Operation) else other
        self.__operations = operations
        self.__feed = feed

    def __repr__(self):
        pipeline = [self.source] + self.segments
        string = ','.join(list(map(repr, pipeline)))
        return f"{self.__class__.__name__}[{string}]"

    def __add__(self, operation):
        operations = self.operations + [operation]
        return Process(self.source, operations)

    def __call__(self, *args, **kwargs):
        pass

    @property
    def operations(self): return self.__operations
    @property
    def feed(self): return self.__feed


class Stage(Logging, ABC):
    def __init__(self, routine, *args, **kwargs):
        Logging.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def signature(self): return list(inspect.signature(self.execute).keys())
    @property
    def parameters(self): return self.signature[self.signature.index("args")+1:self.signature.index("kwargs")]
    @property
    def arguments(self): return self.signature[1:self.signature.index("args")]


class Feed(Logging):
    def __init__(self, *args, inlet=[], **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(inlet, list)
        self.__inlet = inlet

    def __add__(self, operation):
        assert isinstance(operation, Operation)
        return Process(self, operation)

    def execute(self, *args, **kwargs):
        pass

    @property
    def inlet(self): return self.__inlet


class Operation(Logging):
    def __init__(self, *args, inlet=[], outlet=[], **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(inlet, list) and isinstance(outlet, list)
        self.__outlet = outlet
        self.__inlet = inlet

    def execute(self, *args, **kwargs):
        pass

    @property
    def outlet(self): return self.__outlet
    @property
    def inlet(self): return self.__inlet




