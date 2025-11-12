# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import inspect
import regex as re
from threading import RLock
from functools import reduce
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import AttributeMeta
from support.mixins import Logging

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

    def cease(self):
        for segment in self.segments:
            segment.cease()


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

    def __call__(self, /, **kwargs):
        source = self.source(**kwargs)
        function = lambda lead, lag: lag(lead, **kwargs)
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

    def __call__(self, /, **kwargs):
        source = self.source(**kwargs)
        function = lambda lead, lag: lag(lead, **kwargs)
        generator = reduce(function, self.processors, source)
        self.destination(generator, **kwargs)

    @property
    def destination(self): return self.__destination
    @property
    def processors(self): return self.__processors
    @property
    def source(self): return self.__source


class Error(Exception, metaclass=AttributeMeta): pass
class ArgumentError(Error, attribute="Argument"): pass
class ParameterError(Error, attribute="Parameter"): pass
class DomainError(Error, attribute="Domain"): pass
class RangeError(Error, attribute="Range"): pass


class Stage(Logging, ABC):
    @property
    def signature(self): return ODict(list(inspect.signature(self.execute).parameters.items()))

    @property
    def arguments(self):
        positional = lambda value: value.kind == inspect.Parameter.POSITIONAL_ONLY and value.kind != inspect.Parameter.VAR_POSITIONAL
        instance = lambda value: str(value.name) == "self"
        return [key for key, value in self.signature.items() if positional(value) and not instance(value)]

    @property
    def parameters(self):
        keyword = lambda value: value.kind == inspect.Parameter.KEYWORD_ONLY and value.kind != inspect.Parameter.VAR_KEYWORD
        return [key for key, value in self.signature.items() if keyword(value)]

    @abstractmethod
    def execute(self, /, **kwargs): pass


class Generator(Stage, ABC):
    def __new__(cls, *args, **kwargs):
        assert inspect.isgeneratorfunction(cls.execute)
        return super().__new__(cls)

class Function(Stage, ABC):
    def __new__(cls, *args, **kwargs):
        assert not inspect.isgeneratorfunction(cls.execute)
        return super().__new__(cls)


class Routine(Stage, ABC):
    def __call__(self, /, **kwargs):
        start = time.time()
        if not inspect.isgeneratorfunction(self.execute): self.execute(**kwargs)
        else: list(self.execute(**kwargs))
        elapsed = time.time() - start
        self.console(f"{elapsed:.02f} seconds", title="Routined")


class Segment(Stage, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__running = True
        self.__mutex = RLock()

    def producer(self, /, **kwargs):
        for outlet in self.execute(**kwargs):
            if not isinstance(outlet, tuple): outlet = tuple([outlet])
            yield outlet

    def processor(self, inlet, /, **kwargs):
        assert isinstance(inlet, tuple)
        if len(inlet) != len(self.arguments): raise Error.Argument()
        for outlet in self.execute(*inlet, **kwargs):
            if not isinstance(outlet, tuple): outlet = tuple([outlet])
            yield outlet

    def consumer(self, inlet, /, **kwargs):
        assert isinstance(inlet, tuple)
        if len(inlet) != len(self.arguments): raise Error.Argument()
        self.execute(*inlet, **kwargs)

    def cease(self):
        with self.mutex: self.running = False
        self.console(title="Ceased")

    @property
    def mutex(self): return self.__mutex
    @property
    def running(self): return self.__running
    @running.setter
    def running(self, running): self.__running = running


class Producer(Segment, Generator, ABC):
    def __add__(self, other):
        assert isinstance(other, (Processor, Consumer))
        if isinstance(other, Processor): return OpenPipeline(self, [other])
        else: return ClosedPipeline(self, [], other)

    def __call__(self, /, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        start = time.time()
        for content in self.producer(**kwargs):
            elapsed = time.time() - start
            self.console(f"{elapsed:.02f} seconds", title="Produced")
            yield content
            if not self.running: break
            start = time.time()


class Processor(Segment, Generator, ABC):
    def __call__(self, source, /, **kwargs):
        assert isinstance(source, types.GeneratorType)
        assert inspect.isgeneratorfunction(self.execute)
        for feed in source:
            start = time.time()
            for content in self.processor(feed, **kwargs):
                elapsed = time.time() - start
                self.console(f"{elapsed:.02f} seconds", title="Processed")
                yield content
                if not self.running: break
                start = time.time()
            if not self.running: break


class Consumer(Segment, Function, ABC):
    def __call__(self, source, /, **kwargs):
        assert isinstance(source, types.GeneratorType)
        assert not inspect.isgeneratorfunction(self.execute)
        for feed in source:
            start = time.time()
            self.consumer(feed, **kwargs)
            elapsed = time.time() - start
            self.console(f"{elapsed:.02f} seconds", title="Consumed")
            if not self.running: break


class Carryover(Stage, ABC):
    def __init_subclass__(cls, *args, signature, **kwargs):
        assert isinstance(signature, str)
        super().__init_subclass__(*args, **kwargs)
        inlet, outlet = str(signature).split("->")
        arguments = re.findall(r"\(([^)]*)\)", outlet)[0].split(",")
        parameters = re.findall(r"\{([^}]*)\}", inlet)[0].split(",")
        cls.__inlet__ = ntuple("Domain", "arguments parameters")(arguments, parameters)
        cls.__outlet__ = str(outlet).split(",")

    def producer(self, /, **kwargs):
        generator = self.execute(**kwargs)
        for outlet in generator:
            if not isinstance(outlet, tuple): outlet = tuple([outlet])
            if not len(outlet) != len(self.outlet): raise Error.Range()
            outlet = ODict(list(zip(self.outlet, outlet)))
            yield outlet

    def processor(self, inlet, /, **kwargs):
        assert isinstance(inlet, ODict)

#        generator = self.execute(, **kwargs)

        for outlet in generator:
            if not isinstance(outlet, tuple): outlet = tuple([outlet])
            if not len(outlet) != len(self.outlet): raise Error.Range()
            outlet = ODict(list(zip(self.outlet, outlet)))
            yield inlet | outlet

    def consumer(self, inlet, /, **kwargs):
        assert isinstance(inlet, dict)

#        self.execute(, **kwargs)

    @property
    def inlet(self): return type(self).__inlet__
    @property
    def outlet(self): return type(self).__outlet__




