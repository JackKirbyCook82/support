# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import logging
import inspect
from functools import reduce
from abc import ABC, ABCMeta, abstractmethod

from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Producer", "CycleProducer", "CycleBreaker", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class PipelineMeta(ABCMeta):
    def __call__(cls, feed, stage):
        assert isinstance(feed, (OpenPipeline, Producer))
        assert isinstance(stage, (Processor, Consumer))
        producer = feed.producer if isinstance(feed, OpenPipeline) else feed
        processors = feed.processors if isinstance(feed, OpenPipeline) else []
        processors = processors + ([stage] if isinstance(stage, Processor) else [])
        consumer = stage if isinstance(stage, Consumer) else None
        if consumer is not None:
            instance = super(PipelineMeta, ClosedPipeline).__call__(producer, processors, consumer)
        else:
            instance = super(PipelineMeta, OpenPipeline).__call__(producer, processors)
        return instance


class Pipeline(ABC, metaclass=PipelineMeta):
    def __repr__(self): return "|".join([repr(stage) for stage in self.stages])
    def __init__(self, producer): self.__producer = producer
    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    @property
    def producer(self): return self.__producer


class OpenPipeline(Pipeline):
    def __init__(self, producer, processors):
        assert isinstance(producer, Producer)
        assert isinstance(processors, list)
        assert all([isinstance(processor, Processor) for processor in processors])
        super().__init__(producer)
        self.__processors = processors

    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer))
        return Pipeline(self, stage)

    def execute(self, *args, **kwargs):
        producer = self.producer(*args, **kwargs)
        assert isinstance(producer, types.GeneratorType)
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), self.processors, producer)
        yield from iter(generator)

    @property
    def processors(self): return self.__processors


class ClosedPipeline(OpenPipeline):
    def __init__(self, producer, processors, consumer):
        assert isinstance(consumer, Consumer)
        super().__init__(producer, processors)
        self.__consumer = consumer

    def execute(self, *args, **kwargs):
        producer = self.producer(*args, **kwargs)
        assert isinstance(producer, types.GeneratorType)
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), self.processors, producer)
        return self.consumer(generator, *args, **kwargs)

    @property
    def consumer(self): return self.__consumer


class Stage(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __init__(self, *args, **kwargs):
        self.__title = kwargs.get("title", self.__class__.__title__)
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __repr__(self): return self.name
    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name
    @property
    def title(self): return self.__title


class Generator(Stage, ABC, title="Generated"):
    def generator(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        start = time.time()
        for content in iter(generator):
            LOGGER.info(f"{self.title}: {repr(self)}[{time.time() - start:.2f}s]")
            yield content
            start = time.time()


class Producer(Generator, ABC, title="Produced"):
    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer))
        return Pipeline(self, stage)

    @staticmethod
    def prepare(*args, **kwargs): return {}
    def process(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        parameters = self.prepare(*args, **kwargs)
        kwargs = kwargs | parameters
        generator = self.generator(*args, **kwargs)
        yield from generator


class CycleBreaker(object, metaclass=SingletonMeta):
    def __bool__(self): return self.state
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__state = True

    @property
    def state(self): return self.__state
    @state.setter
    def state(self, state): self.__state = state
    @property
    def name(self): return self.__name


class CycleProducer(Producer, ABC):
    def __init__(self, *args, breaker, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(breaker, CycleBreaker)
        self.__breaker = breaker

    def process(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        parameters = self.prepare(*args, **kwargs)
        kwargs = kwargs | parameters
        while bool(self.breaker):
            generator = self.generator(*args, **kwargs)
            yield from generator

    @property
    def breaker(self): return self.__breaker


class Processor(Generator, ABC, title="Processed"):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        assert inspect.isgeneratorfunction(self.execute)
        for query in iter(stage):
            generator = self.generator(query, *args, **kwargs)
            yield from generator


class Consumer(Stage, ABC, title="Consumed"):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        assert not inspect.isgeneratorfunction(self.execute)
        for query in iter(stage):
            start = time.time()
            self.execute(query, *args, **kwargs)
            LOGGER.info(f"{self.title}: {repr(self)}[{time.time() - start:.2f}s]")






