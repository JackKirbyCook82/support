# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import logging
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Stage", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Stage(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__formatter__ = kwargs.get("formatter", getattr(cls, "__formatter__", None))
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__formatter = kwargs.get("formatter", self.__class__.__formatter__)
        self.__title = kwargs.get("title", self.__class__.__title__)
        self.__name = kwargs.get("name", self.__class__.__name__)

#    @abstractmethod
#    def process(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name
    @property
    def title(self): return self.__title
    @property
    def formatter(self): return self.__formatter


class PipelineMeta(ABCMeta):
    def __call__(cls, *stages):
        producer, stages = stages[0], stages[1:]
        if not bool(stages):
            return OpenPipeline(producer)
        elif not any([isinstance(stage, Consumer) for stage in stages]):
            return OpenPipeline(producer, *stages)
        else:
            return ClosedPipeline(producer, *stages)


class Pipeline(tuple, ABC):
    def __new__(cls, *stages):
        if issubclass(cls, Pipeline):
            return super().__new__(cls, *stages)
        assert len(stages) >= 2
        producer, stages = stages[0], stages[1:]
        if not any([isinstance(stage, Consumer) for stage in stages]):
            assert all([isinstance(stage, Processor) for stage in stages])
            return OpenPipeline(producer, *stages)
        stages, consumer = stages[:-1], stages[-1]
        assert all([isinstance(stage, Processor) for stage in stages])
        return ClosedPipeline(producer, *stages, consumer)


class OpenPipeline(Pipeline):
    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer))
        return Pipeline(*self, stage)


class ClosedPipeline(Pipeline):
    pass


class Producer(Stage, title="Producer"):
    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer))
        return Pipeline(self, stage)

    @abstractmethod
    def producer(self, *args, **kwargs): pass


class Processor(Stage, title="Processed"):
    @abstractmethod
    def processor(self, *args, **kwargs): pass


class Consumer(Stage, title="Consumed"):
    @abstractmethod
    def consumer(self, *args, **kwargs): pass








class Generator(Stage, ABC, title="Generated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__stage = None

    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer)) and stage is not self
        self.stage = stage if self.terminal else (self.stage + stage)
        return self

    def __call__(self, *args, **kwargs):
        generator = self.process(*args, **kwargs)
        if self.terminal:
            yield from generator
        elif isinstance(self.stage, Generator):
            generator = self.stage(generator, *args, **kwargs)
            yield from generator
        else:
            self.stage(generator, *args, **kwargs)
            return

    def generator(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        start = time.time()
        for query in iter(generator):
            assert isinstance(query, dict)
            elapsed = time.time() - start
            string = self.formatter(self, *args, query=query, elapsed=elapsed, **kwargs)
            __logger__.info(string)
            yield query
            start = time.time()

    @property
    def terminal(self): return bool(self.stage is None)
    @property
    def termination(self): return self if self.terminal else self.stage.termination
    @property
    def open(self): return isinstance(self.termination, (Producer, Processor))
    @property
    def close(self): return isinstance(self.termination, Consumer)

    @property
    def stage(self): return self.__stage
    @stage.setter
    def stage(self, stage): self.__stage = stage


class Producer(Generator, ABC, title="Produced"):
    def process(self, *args, **kwargs):
        generator = self.generator(*args, **kwargs)
        yield from generator


class Processor(Generator, ABC, title="Processed"):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for query in iter(stage):
            generator = self.generator(query, *args, **kwargs)
            yield from generator


class Consumer(Stage, ABC, title="Consumed"):
    def __call__(self, *args, **kwargs):
        self.process(*args, **kwargs)

    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for query in iter(stage):
            assert isinstance(query, dict)
            start = time.time()
            self.execute(query, *args, **kwargs)
            elapsed = time.time() - start
            string = self.formatter(*args, query=query, elapsed=elapsed, **kwargs)
            __logger__.info(string)



