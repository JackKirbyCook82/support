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


class Pipeline(ABC):
    def __repr__(self): return f"{type(self).__name__}[{','.join(list(map(lambda segment: segment.name, self.segments)))}]"
    def __init__(self, segments): self.__segments = segments
    def __getitem__(self, index): return self.segments[index]

    @property
    def segments(self): return self.__segments


class OpenPipeline(Pipeline, ABC):
    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if not isinstance(other, Consumer):
            parameters = dict(producer=self.producer, processors=list(self.processors) + [other])
            return OpenPipeline(**parameters)
        else:
            parameters = dict(producer=self.producer, processors=list(self.processors), consumer=other)
            return ClosedPipeline(**parameters)

    def __init__(self, *args, producer, processors=[], **kwargs):
        assert isinstance(processors, list)
        super().__init__([producer] + processors)
        self.__producer = producer
        self.__processors = processors

    @property
    def producer(self): return self.__producer
    @property
    def processors(self): return self.__processors


class ClosedPipeline(Pipeline):
    def __init__(self, *args, producer, processors=[], consumer, **kwargs):
        assert isinstance(processors, list)
        super().__init__([producer] + processors + [consumer])
        self.__producer = producer
        self.__processors = processors
        self.__consumer = consumer

    def __call__(self, *args, **kwargs):
        function = lambda lead, lag: lag(lead, *args, **kwargs)
        generator = self.producer(*args, **kwargs)
        generator = reduce(function, list(self.processors), generator)
        self.consumer(generator, *args, **kwargs)

    @property
    def producer(self): return self.__producer
    @property
    def processors(self): return self.__processors
    @property
    def consumer(self): return self.__consumer


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
    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if not isinstance(other, Consumer):
            return OpenPipeline(producer=self, processors=[other])
        else:
            return ClosedPipeline(producer=self, consumer=other)

    def __call__(self, *args, **kwargs):
        assert not bool(args)
        start = time.time()
        for results in self.producer(*args, **kwargs):
            assert isinstance(results, dict)
            elapsed = time.time() - start
            parameters = dict(query=results, elapsed=elapsed)
            string = self.formatter(self, **parameters)
            __logger__.info(string)
            yield results
            start = time.time()

    @abstractmethod
    def producer(self, *args, **kwargs): pass


class Processor(Stage, title="Processed"):
    def __call__(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for contents in source:
            start = time.time()
            for results in self.processor(contents, *args, **kwargs):
                assert isinstance(results, dict)
                elapsed = time.time() - start
                parameters = dict(query=results, elapsed=elapsed)
                string = self.formatter(self, **parameters)
                __logger__.info(string)
                yield results
                start = time.time()

    @abstractmethod
    def processor(self, contents, *args, **kwargs): pass


class Consumer(Stage, title="Consumed"):
    def __call__(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for contents in source:
            start = time.time()
            self.consumer(contents, *args, **kwargs)
            elapsed = time.time() - start
            parameters = dict(elapsed=elapsed)
            string = self.formatter(self, **parameters)
            __logger__.info(string)

    @abstractmethod
    def consumer(self, contents, *args, **kwargs): pass



