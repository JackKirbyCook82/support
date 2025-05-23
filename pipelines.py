# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import inspect
from threading import RLock
from functools import reduce
from abc import ABC, abstractmethod

from support.mixins import Function, Generator, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Routine", "Producer", "Processor", "Consumer", "Carryover"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Pipeline(ABC):
    def __init__(self, segments): self.segments = segments
    def __repr__(self):
        string = ', '.join(list(map(repr, self.segments)))
        return f"{self.__class__.__name__}[{string}]"

    def cease(self, *args, **kwargs):
        for segment in self.segments:
            segment.cease(*args, **kwargs)


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


class Routine(Stage, ABC):
    def __call__(self, *args, **kwargs):
        start = time.time()
        if not inspect.isgeneratorfunction(self.execute): self.execute(*args, **kwargs)
        else: list(self.execute(*args, **kwargs))
        elapsed = time.time() - start
        self.console(f"{elapsed:.02f} seconds", title="Routined")


class Segment(Stage, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__running = True
        self.__mutex = RLock()

    def producer(self, *args, **kwargs):
        for content in self.execute(*args, **kwargs):
            content = [content] if not isinstance(content, tuple) else list(content)
            yield content

    def processor(self, feed, *args, **kwargs):
        assert isinstance(feed, tuple)
        for content in self.execute(*feed, *args, **kwargs):
            content = [content] if not isinstance(content, tuple) else list(content)
            yield content

    def consumer(self, feed, *args, **kwargs):
        assert isinstance(feed, tuple)
        self.execute(*feed, *args, **kwargs)

    def cease(self, *args, **kwargs):
        with self.mutex: self.running = False
        self.console(title="Ceased")

    @property
    def mutex(self): return self.__mutex
    @property
    def running(self): return self.__running
    @running.setter
    def running(self, running): self.__running = running


class Producer(Generator, Segment, ABC):
    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if isinstance(other, Processor): return OpenPipeline(self, [other])
        else: return ClosedPipeline(self, [], other)

    def __call__(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        start = time.time()
        for content in self.producer(*args, **kwargs):
            elapsed = time.time() - start
            self.console(f"{elapsed:.02f} seconds", title="Produced")
            yield content
            if not self.running: break
            start = time.time()


class Processor(Generator, Segment, ABC):
    def __call__(self, source, *args, **kwargs):
        assert isinstance(source, types.GeneratorType)
        assert inspect.isgeneratorfunction(self.execute)
        for feed in source:
            start = time.time()
            for content in self.processor(feed, *args, **kwargs):
                elapsed = time.time() - start
                self.console(f"{elapsed:.02f} seconds", title="Processed")
                yield content
                if not self.running: break
                start = time.time()
            if not self.running: break


class Consumer(Function, Segment, ABC):
    def __call__(self, source, *args, **kwargs):
        assert isinstance(source, types.GeneratorType)
        assert not inspect.isgeneratorfunction(self.execute)
        for feed in source:
            start = time.time()
            self.consumer(feed, *args, **kwargs)
            elapsed = time.time() - start
            self.console(f"{elapsed:.02f} seconds", title="Consumed")
            if not self.running: break


class Carryover(Stage, ABC):
    def __init_subclass__(cls, *args, signature="", **kwargs):
        assert isinstance(signature, str)
        super().__init_subclass__(*args, **kwargs)
        if not bool(signature): return
        assert "->" in str(signature)
        inlet, outlet = str(signature).split("->")
        cls.__domain__ = list(filter(bool, str(inlet).split(",")))
        cls.__range__ = list(filter(bool, str(outlet).split(",")))

    def producer(self, *args, **kwargs):
        for content in self.execute(*args, **kwargs):
            content = [content] if not isinstance(content, tuple) else list(content)
            assert len(self.range) == len(content)
            content = dict(zip(self.range, content))
            yield content

    def processor(self, feed, *args, **kwargs):
        assert isinstance(feed, dict)
        assert not set(self.domain) - set(feed.keys())
        domain = [feed[key] for key in self.domain]
        for content in self.execute(*domain, *args, **kwargs):
            content = [content] if not isinstance(content, tuple) else list(content)
            assert len(self.range) == len(content)
            content = dict(zip(self.range, content))
            yield feed | content

    def consumer(self, feed, *args, **kwargs):
        assert isinstance(feed, dict)
        assert not set(self.domain) - set(feed.keys())
        feed = [feed[key] for key in self.domain]
        self.execute(*feed, *args, **kwargs)

    @property
    def domain(self): return type(self).__domain__
    @property
    def range(self): return type(self).__range__



