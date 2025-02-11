# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import inspect
from functools import reduce
from abc import ABC, abstractmethod

from support.mixins import Function, Generator, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Routine", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Pipeline(ABC):
    def __init__(self, segments): self.segments = segments
    def __repr__(self):
        string = ', '.join(list(map(repr, self.segments)))
        return f"{self.__class__.__name__}[{string}]"


class OpenPipeline(Pipeline):
    def __init__(self, source, processors):
        assert isinstance(source, Producer) and isinstance(processors, list)
        assert all([isinstance(processor, Processor) for processor in processors])
        super().__init__([source] + processors)
        self.__processors = processors
        self.__source = source

    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if isinstance(other, Processor): return OpenPipeline(self.source, self.processors + [other])
        else: return ClosedPipeline(self.source, self.processors, other)

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        function = lambda lead, lag: lag(lead, *args, **kwargs)
        generator = reduce(function, self.processors, source)
        yield from generator

    @property
    def processors(self): return self.__processors
    @property
    def source(self): return self.__source


class ClosedPipeline(Pipeline):
    def __init__(self, source, processors, destination):
        assert isinstance(destination, Consumer) and isinstance(processors, list)
        assert all([isinstance(processor, Processor) for processor in processors])
        super().__init__([source] + processors + [destination])
        self.__destination = destination
        self.__processors = processors
        self.__source = source

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        function = lambda lead, lag: lag(lead, *args, **kwargs)
        generator = reduce(function, self.processors, source)
        self.destination(generator, *args, **kwargs)

    @property
    def destination(self): return self.__destination
    @property
    def processors(self): return self.__processors
    @property
    def source(self): return self.__source


class Stage(Logging, ABC):
    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Source(Stage, ABC):
    def generator(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        source = self.execute(*args, **kwargs)
        start = time.time()
        for content in source:
            elapsed = time.time() - start
            self.console(f"{elapsed:.02f} sec", title="Produced")
            yield content
            start = time.time()


class Routine(Stage, ABC):
    def __call__(self, *args, **kwargs):
        start = time.time()
        if not inspect.isgeneratorfunction(self.execute): self.execute(*args, **kwargs)
        else: list(self.execute(*args, **kwargs))
        elapsed = time.time() - start
        self.console(f"{elapsed:.02f} sec", title="Routined")


class Producer(Generator, Source, ABC):
    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if isinstance(other, Processor): return OpenPipeline(self, [other])
        else: return ClosedPipeline(self, [], other)

    def __call__(self, *args, **kwargs):
        generator = self.generator(*args, **kwargs)
        yield from generator


class Processor(Generator, Source, ABC):
    def __call__(self, source, *args, **kwargs):
        assert isinstance(source, types.GeneratorType)
        for content in source:
            if not isinstance(content, tuple): generator = self.generator(content, *args, **kwargs)
            else: generator = self.generator(*content, *args, **kwargs)
            yield from generator


class Consumer(Function, Stage, ABC):
    def __call__(self, source, *args, **kwargs):
        assert isinstance(source, types.GeneratorType)
        for content in source:
            start = time.time()
            assert not inspect.isgeneratorfunction(self.execute)
            self.execute(content, *args, **kwargs)
            elapsed = time.time() - start
            self.console(f"{elapsed:.02f} sec", title="Consumed")



