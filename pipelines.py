# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import inspect
import logging
from functools import reduce
from abc import ABC, abstractmethod

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Stage", "Routine", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Pipeline(ABC):
    def __init__(self, segments):
        assert inspect.isgeneratorfunction(self.execute)
        self.__name = self.__class__.__name__
        self.__segments = segments

    def __repr__(self):
        function = lambda segment: segment.name
        string = ','.join(list(map(function, self.segments)))
        return f"{self.name}[{string}]"

    def __getitem__(self, index): return self.segments[index]
    def __call__(self, *args, **kwargs):
        execute = self.execute(*args, **kwargs)
        assert isinstance(execute, types.GeneratorType)
        yield from execute

    def generator(self, *args, **kwargs):
        function = lambda lead, lag: lag(lead, *args, **kwargs)
        generator = self.producer(*args, **kwargs)
        generator = reduce(function, list(self.processors), generator)
        return generator

    @property
    @abstractmethod
    def producer(self): pass
    @property
    @abstractmethod
    def processors(self): pass

    @property
    def segments(self): return self.__segments
    @property
    def name(self): return self.__name


class OpenPipeline(Pipeline):
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

    def execute(self, *args, **kwargs):
        generator = self.generator(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        yield from generator

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

    def execute(self, *args, **kwargs):
        generator = self.generator(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        self.consumer(generator, *args, **kwargs)
        return
        yield

    @property
    def producer(self): return self.__producer
    @property
    def processors(self): return self.__processors
    @property
    def consumer(self): return self.__consumer


class Stage(Logging, ABC):
    def __call__(self, *args, **kwargs):
        if not inspect.isgeneratorfunction(self.execute): yield self.execute(*args, **kwargs)
        else: yield from self.execute(*args, **kwargs)

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Routine(Stage):
    def __init__(self, routine, *args, **kwargs):
        assert callable(routine)
        assert not inspect.isgeneratorfunction(self.execute)
        assert not inspect.isgeneratorfunction(routine)
        Stage.__init__(self, *args, **kwargs)
        self.routine = routine

    def execute(self, *args, **kwargs):
        assert not bool(args)
        start = time.time()
        self.routine(*args, **kwargs)
        elapsed = (time.time() - start).total_seconds()
        string = f"Performed: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)


class Producer(Stage, ABC):
    def __init__(self, producer, *args, **kwargs):
        assert callable(producer)
        assert inspect.isgeneratorfunction(self.execute)
        assert inspect.isgeneratorfunction(producer)
        Stage.__init__(self, *args, **kwargs)
        self.producer = producer

    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if not isinstance(other, Consumer): return OpenPipeline(producer=self, processors=[other])
        else: return ClosedPipeline(producer=self, consumer=other)

    def execute(self, *args, **kwargs):
        assert not bool(args)
        start = time.time()
        for produced in self.producer(*args, **kwargs):
            assert isinstance(produced, dict)
            elapsed = time.time() - start
            string = f"Produced: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)
            yield produced
            start = time.time()


class Processor(Stage, ABC):
    def __init__(self, processor, *args, **kwargs):
        assert callable(processor)
        assert inspect.isgeneratorfunction(self.execute)
        assert inspect.isgeneratorfunction(processor)
        Stage.__init__(self, *args, **kwargs)
        self.processor

    def execute(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for consumed in source:
            start = time.time()
            for produced in self.processor(consumed, *args, **kwargs):
                assert isinstance(produced, dict)
                elapsed = time.time() - start
                string = f"Processed: {repr(self)}[{elapsed:.02f}s]"
                self.logger.info(string)
                yield produced
                start = time.time()


class Consumer(Stage, ABC):
    def __init__(self, consumer, *args, **kwargs):
        assert callable(consumer)
        assert not inspect.isgeneratorfunction(self.execute)
        assert not inspect.isgeneratorfunction(consumer)
        Stage.__init__(self, *args, **kwargs)
        self.consumer = consumer

    def execute(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for consumed in source:
            start = time.time()
            self.consumer(consumed, *args, **kwargs)
            elapsed = time.time() - start
            string = f"Consumed: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)


