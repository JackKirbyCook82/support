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

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Producer", "Processor", "Consumer", "Reader", "Writer", "Stack"]
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
        producer, processors = self.stages[0](*args, **kwargs), self.stages[1:]
        assert isinstance(producer, types.GeneratorType)
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), processors, producer)
        yield from iter(generator)

    @property
    def stages(self): return [self.producer]
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

    @property
    def stages(self): return super().stages + list(self.processors)
    @property
    def processors(self): return self.__processors


class ClosedPipeline(OpenPipeline):
    def __init__(self, producer, processors, consumer):
        assert isinstance(consumer, Consumer)
        super().__init__(producer, processors)
        self.__consumer = consumer

    @property
    def stages(self): return super().stages + [self.consumer]
    @property
    def consumer(self): return self.__consumer


class Stage(ABC):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs): pass
    @abstractmethod
    def generator(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name


class Stack(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__type__ = kwargs.get("type", getattr(cls, "__type__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        stackname = kwargs.get("name", self.__class__.__name__)
        stacktype = self.__class__.__type__
        assert stacktype is not None
        self.__name = stackname
        self.__type = stacktype

    @abstractmethod
    def read(self, *args, **kwargs): pass
    @abstractmethod
    def write(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name
    @property
    def type(self): return self.__type


class Producer(Stage, ABC):
    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer))
        return Pipeline(self, stage)

    def process(self, *args, **kwargs):
        generator = self.generator(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        return generator

    def generator(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        start = time.time()
        generator = self.execute(*args, **kwargs)
        for content in iter(generator):
            LOGGER.info("Produced: {}[{:.2f}s]".format(repr(self), time.time() - start))
            yield content
            start = time.time()


class Processor(Stage, ABC):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for content in iter(stage):
            generator = self.generator(content, *args, **kwargs)
            assert isinstance(generator, types.GeneratorType)
            yield from iter(generator)

    def generator(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        start = time.time()
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        for content in iter(generator):
            LOGGER.info("Processed: {}[{:.2f}s]".format(repr(self), time.time() - start))
            yield content
            start = time.time()


class Consumer(Stage, ABC):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for content in iter(stage):
            generator = self.generator(content, *args, **kwargs)
            assert isinstance(generator, types.GeneratorType)
            yield from iter(generator)

    def generator(self, *args, **kwargs):
        assert not inspect.isgeneratorfunction(self.execute)
        start = time.time()
        self.execute(*args, **kwargs)
        LOGGER.info("Consumed: {}[{:.2f}s]".format(repr(self), time.time() - start))
        return
        yield


class Reader(Producer, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    def read(self, *args, **kwargs):
        return self.source.read(*args, **kwargs)

    @property
    def source(self): return self.__source


class Writer(Consumer, ABC):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.__destination = destination

    def write(self, *args, **kwargs):
        self.destination.write(*args, **kwargs)

    @property
    def destination(self): return self.__destination







