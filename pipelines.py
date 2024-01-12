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
__all__ = ["Producer", "Processor", "Consumer", "Reader", "Writer"]
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
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name


class Producer(Stage, ABC):
    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer))
        return Pipeline(self, stage)

    def process(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        start = time.time()
        for content in iter(generator):
            LOGGER.info(f"Produced: {repr(self)}[{time.time() - start:.2f}s]")
            yield content
            start = time.time()


class Processor(Stage, ABC):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for query in iter(stage):
            assert inspect.isgeneratorfunction(self.execute)
            generator = self.execute(query, *args, **kwargs)
            assert isinstance(generator, types.GeneratorType)
            start = time.time()
            for content in iter(generator):
                LOGGER.info(f"Processed: {repr(self)}[{time.time() - start:.2f}s]")
                yield content
                start = time.time()


class Consumer(Stage, ABC):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for query in iter(stage):
            assert not inspect.isgeneratorfunction(self.execute)
            start = time.time()
            self.execute(query, *args, **kwargs)
            LOGGER.info(f"Consumed: {repr(self)}[{time.time() - start:.2f}s]")


class Reader(Producer, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    @property
    def source(self): return self.__source


class Writer(Consumer, ABC):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.__destination = destination

    @property
    def destination(self): return self.__destination



