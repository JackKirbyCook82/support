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

from support.mixins import Mixin

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Stage", "Routine", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


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
        cls.__reporting__ = kwargs.get("reporting", getattr(cls, "__reporting__", False))
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__reporting = kwargs.get("reporting", self.__class__.__reporting__)
        self.__variable = kwargs.get("variable", self.__class__.__variable__)
        self.__title = kwargs.get("title", self.__class__.__title__)
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def report(self, *args, variable=None, elapsed, **kwargs):
        if not bool(self.reporting): return
        if variable is None: string = f"{str(self.title)}: {repr(self)}[{elapsed:.02f}s]"
        else: string = f"{str(self.title)}: {repr(self)}|{str(variable)}[{elapsed:.02f}s]"
        __logger__.info(string)

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def reporting(self): return self.__reporting
    @property
    def variable(self): return self.__variable
    @property
    def title(self): return self.__title
    @property
    def name(self): return self.__name


class Routine(Stage, ABC, title="Routined"):
    def execute(self, *args, **kwargs):
        assert not bool(args)
        start = time.time()
        self.routine(*args, **kwargs)
        elapsed = time.time() - start
        self.report(elapsed=elapsed)

    @abstractmethod
    def routine(self, *args, **kwargs): pass


class Producer(Stage, ABC, title="Produced"):
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
            variable = produced[self.variable]
            self.report(variable=variable, elapsed=elapsed)
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
                assert produced[self.variable] == consumed[self.variable]
                variable = produced[self.variable]
                self.report(variable=variable, elapsed=elapsed)
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
            variable = consumed[self.variable]
            self.report(variable=variable, elapsed=elapsed)

    @abstractmethod
    def consumer(self, contents, *args, **kwargs): pass


