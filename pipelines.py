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

from support.mixins import Mixin

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


class Stage(Mixin, ABC):
    def __init_subclass__(cls, **kwargs):
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__title = self.__class__.__title__
        self.__logger = __logger__

    def __repr__(self): return f"{str(self.name)}"
    def __call__(self, *args, **kwargs):
        if not inspect.isgeneratorfunction(self.wrapper): yield self.wrapper(*args, **kwargs)
        else: yield from self.wrapper(*args, **kwargs)

    @abstractmethod
    def wrapper(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


class Routine(Stage, title="Performed"):
    def __init__(self, *args, **kwargs):
        assert not inspect.isgeneratorfunction(self.execute)
        assert not inspect.isgeneratorfunction(self.routine)
        super().__init__(*args, **kwargs)

    def wrapper(self, *args, **kwargs):
        assert not bool(args)
        start = time.time()
        self.execute(*args, **kwargs)
        elapsed = (time.time() - start).total_seconds()
        string = f"{str(self.title).title()}: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)

    @abstractmethod
    def routine(self, *args, **kwargs): pass


class Producer(Stage, ABC, title="Produced"):
    def __init__(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        assert inspect.isgeneratorfunction(self.producer)
        super().__init__(*args, **kwargs)

    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if not isinstance(other, Consumer): return OpenPipeline(producer=self, processors=[other])
        else: return ClosedPipeline(producer=self, consumer=other)

    def wrapper(self, *args, **kwargs):
        assert not bool(args)
        start = time.time()
        for produced in self.execute(*args, **kwargs):
            assert isinstance(produced, dict)
            elapsed = time.time() - start
            string = f"{str(self.title).title()}: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)
            yield produced
            start = time.time()


class Processor(Stage, ABC, title="Processed"):
    def __init__(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        assert inspect.isgeneratorfunction(self.processor)
        super().__init__(*args, **kwargs)

    def wrapper(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for consumed in source:
            start = time.time()
            for produced in self.execute(consumed, *args, **kwargs):
                assert isinstance(produced, dict)
                elapsed = time.time() - start
                string = f"{str(self.title).title()}: {repr(self)}[{elapsed:.02f}s]"
                self.logger.info(string)
                yield produced
                start = time.time()


class Consumer(Stage, ABC, title="Consumed"):
    def __init__(self, *args, **kwargs):
        assert not inspect.isgeneratorfunction(self.execute)
        assert not inspect.isgeneratorfunction(self.consumer)
        super().__init__(*args, **kwargs)

    def wrapper(self, source, *args, **kwargs):
        assert not bool(args)
        assert isinstance(source, types.GeneratorType)
        for consumed in source:
            start = time.time()
            self.execute(consumed, *args, **kwargs)
            elapsed = time.time() - start
            string = f"{str(self.title).title()}: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)


