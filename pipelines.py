# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
from functools import reduce
from abc import ABC, abstractmethod

from support.mixins import Mixin

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Stage", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Pipeline(ABC):
    def __init__(self, segments): self.__segments = segments

    def __repr__(self): return f"{type(self).__name__}[{','.join(list(map(lambda segment: segment.name, self.segments)))}]"
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
        generator = self.generator(*args, **kwargs)
        self.consumer(generator, *args, **kwargs)

    def generator(self, *args, **kwargs):
        function = lambda lead, lag: lag(lead, *args, **kwargs)
        generator = self.producer(*args, **kwargs)
        generator = reduce(function, list(self.processors), generator)
        return generator

    @property
    def producer(self): return self.__producer
    @property
    def processors(self): return self.__processors
    @property
    def consumer(self): return self.__consumer


class Stage(Mixin, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__title = kwargs.get("title", self.__class__.__title__)
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @abstractmethod
    def report(self, *args, consumed, produced, elapsed, **kwargs): pass

    @property
    def title(self): return self.__title
    @property
    def name(self): return self.__name


class Producer(Stage, ABC, title="Producer"):
    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if not isinstance(other, Consumer):
            return OpenPipeline(producer=self, processors=[other])
        else:
            return ClosedPipeline(producer=self, consumer=other)

    def execute(self, *args, **kwargs):
        assert not bool(args)
        start = time.time()
        for produced in self.producer(*args, **kwargs):
            assert isinstance(produced, dict)
            elapsed = time.time() - start
            parameters = dict(produced=produced, elapsed=elapsed)
            self.report(*args, **parameters, **kwargs)
            yield produced
            start = time.time()

    @abstractmethod
    def producer(self, *args, **kwargs): pass


class Processor(Stage, ABC, title="Processed"):
    def execute(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for consumed in source:
            start = time.time()
            for produced in self.processor(consumed, *args, **kwargs):
                assert isinstance(produced, dict)
                elapsed = time.time() - start
                parameters = dict(consumed=consumed, produced=produced, elapsed=elapsed)
                self.report(*args, **parameters, **kwargs)
                yield produced
                start = time.time()

    @abstractmethod
    def processor(self, contents, *args, **kwargs): pass


class Consumer(Stage, ABC, title="Consumed"):
    def execute(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for consumed in source:
            start = time.time()
            self.consumer(consumed, *args, **kwargs)
            elapsed = time.time() - start
            parameters = dict(consumed=consumed, elapsed=elapsed)
            self.report(*args, **parameters, **kwargs)

    @abstractmethod
    def consumer(self, contents, *args, **kwargs): pass



