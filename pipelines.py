# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import os
import time
import types
import logging
import multiprocessing
from functools import reduce
from abc import ABC, ABCMeta, abstractmethod

from support.files import save, load

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Downloader", "Calculator", "Filter", "Loader", "Saver"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class PipelineMeta(ABCMeta):
    def __call__(cls, feed, stage):
        assert isinstance(feed, (OpenPipeline, Source))
        assert isinstance(stage, (Processor, Destination))
        source = feed.source if isinstance(feed, OpenPipeline) else stage
        processors = feed.processors if isinstance(feed, OpenPipeline) else []
        processors = processors + ([stage] if isinstance(stage, Processor) else [])
        destination = stage if isinstance(stage, Destination) else None
        subcls = OpenPipeline if destination is None else ClosedPipeline
        instance = super(PipelineMeta, subcls).__call__(source, processors, destination)
        return instance


class Pipeline(ABC, metaclass=PipelineMeta): pass
class OpenPipeline(Pipeline):
    def __init__(self, source, processors):
        assert isinstance(source, Source)
        assert isinstance(processors, list)
        assert all([isinstance(processor, Processor) for processor in processors])
        self.__source = source
        self.__processors = processors

    def __add__(self, stage):
        assert isinstance(stage, (Processor, Destination))
        return Pipeline(self, stage)

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        assert isinstance(source, types.GeneratorType)
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), self.processors, source)
        yield from iter(generator)

    @property
    def source(self): return self.__source
    @property
    def processors(self): return self.__processors


class ClosedPipeline(OpenPipeline):
    def __init__(self, source, processors, destination):
        super().__init__(source, processors)
        assert isinstance(destination, Destination)
        self.__destination = destination

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        assert isinstance(source, types.GeneratorType)
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), self.processors, source)
        self.destination(generator, *args, **kwargs)

    @property
    def destination(self): return self.__destination


class Stage(ABC):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        generator = self.generator(*args, **kwargs)
        yield from iter(generator)

    @abstractmethod
    def generator(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def name(self): return self.__name


class Source(Stage, ABC):
    def __add__(self, stage):
        assert isinstance(stage, (Processor, Destination))
        return Pipeline(self, stage)

    def generator(self, *args, **kwargs):
        start = time.time()
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        for content in iter(generator):
            LOGGER.info("Source: {}|{:.2f}s".format(repr(self), time.time() - start))
            yield content
            start = time.time()


class Processor(Stage, ABC):
    def generator(self, stage, *args, **kwargs):
        start = time.time()
        assert isinstance(stage, (Source, Processor))
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        for content in iter(generator):
            LOGGER.info("Processor: {}|{:.2f}s".format(repr(self), time.time() - start))
            yield content
            start = time.time()


class Destination(Stage, ABC):
    def generator(self, stage, *args, **kwargs):
        start = time.time()
        assert isinstance(stage, (Source, Processor))
        self.execute(*args, **kwargs)
        LOGGER.info("Destination: {}|{:.2f}s".format(repr(self), time.time() - start))
        return
        yield


class Downloader(Source, ABC): pass
class Calculator(Processor, ABC): pass
class Filter(Processor, ABC): pass

class Loader(Source, ABC): pass
class Saver(Destination, ABC): pass



