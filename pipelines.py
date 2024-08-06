# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import logging
from functools import reduce
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Stage", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Pipeline(object, ABC):
    pass

#    def __new__(cls, *stages):
#        if issubclass(cls, Pipeline):
#            return super().__new__(cls, stages)
#        assert len(stages) >= 2
#        producer, stages = stages[0], stages[1:]
#        if not any([isinstance(stage, Consumer) for stage in stages]):
#            assert all([isinstance(stage, Processor) for stage in stages])
#            return OpenPipeline(producer, *stages)
#        stages, consumer = stages[:-1], stages[-1]
#        assert all([isinstance(stage, Processor) for stage in stages])
#        return ClosedPipeline(producer, *stages, consumer)


class OpenPipeline(Pipeline, ABC):
    pass

#    def __add__(self, stage):
#        assert isinstance(stage, (Processor, Consumer))
#        return Pipeline(*self, stage)


class ClosedPipeline(Pipeline):
    def __call__(self, *args, **kwargs):
        function = lambda lead, lag: lag(lead, *args, **kwargs)
        generator = reduce(function, list(self.processors), self.producer)
        self.consumer(generator, *args, **kwargs)

    @property
    def producer(self): pass
    @property
    def processors(self): pass
    @property
    def consumer(self): pass


class Stage(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__formatter__ = kwargs.get("formatter", getattr(cls, "__formatter__", None))
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__formatter = kwargs.get("formatter", self.__class__.__formatter__)
        self.__title = kwargs.get("title", self.__class__.__title__)
        self.__name = kwargs.get("name", self.__class__.__name__)

    @property
    def formatter(self): return self.__formatter
    @property
    def title(self): return self.__title
    @property
    def name(self): return self.__name


class Producer(Stage, title="Producer"):
#    def __add__(self, stage):
#        assert isinstance(stage, (Processor, Consumer))
#        return Pipeline(self, stage)

    def __call__(self, *args, **kwargs):
        assert not bool(args)
        generator = self.producer(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        start = time.time()
        for contents in generator:
            assert isinstance(contents, dict)
            elapsed = time.time() - start
            parameters = dict(contents=contents, elapsed=elapsed)
            string = self.formatter(self, **parameters)
            __logger__.info(string)
            yield contents
            start = time.time()

    @abstractmethod
    def producer(self, *args, **kwargs): pass


class Processor(Stage, title="Processed"):
    def __call__(self, generator, *args, **kwargs):
        assert not bool(args)
        assert isinstance(generator, types.GeneratorType)
        generator = self.processor(generator, *args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        start = time.time()
        for contents in generator:
            assert isinstance(contents, dict)
            elapsed = time.time() - start
            parameters = dict(contents=contents, elapsed=elapsed)
            string = self.formatter(self, **parameters)
            __logger__.info(string)
            yield contents
            start = time.time()

    @abstractmethod
    def processor(self, generator, *args, **kwargs): pass


class Consumer(Stage, title="Consumed"):
    def __call__(self, generator, *args, **kwargs):
        assert not bool(args)
        assert isinstance(generator, types.GeneratorType)
        for contents in generator:
            assert isinstance(contents, dict)
            start = time.time()
            self.consumer(contents, *args, **kwargs)
            elapsed = time.time() - start
            parameters = dict(contents=contents, elapsed=elapsed)
            string = self.formatter(self, **parameters)
            __logger__.info(string)

    @abstractmethod
    def consumer(self, generator, *args, **kwargs): pass



