# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import os
import time
import types
import queue
import logging
import inspect
from functools import reduce
from abc import ABC, ABCMeta, abstractmethod

from support.files import load, save

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Downloader", "Calculator", "Filter", "Loader", "Saver", "Producer", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class PipelineMeta(ABCMeta):
    def __call__(cls, feed, stage):
        assert isinstance(feed, (OpenPipeline, Source))
        assert isinstance(stage, (Processor, Destination))
        source = feed.source if isinstance(feed, OpenPipeline) else feed
        processors = feed.processors if isinstance(feed, OpenPipeline) else []
        processors = processors + ([stage] if isinstance(stage, Processor) else [])
        destination = stage if isinstance(stage, Destination) else None
        if destination is not None:
            instance = super(PipelineMeta, ClosedPipeline).__call__(source, processors, destination)
        else:
            instance = super(PipelineMeta, OpenPipeline).__call__(source, processors)
        return instance


class Pipeline(ABC, metaclass=PipelineMeta):
    def __repr__(self): return "|".format([repr(stage) for stage in self.stages])
    def __init__(self, source): self.__source = source

    def __call__(self, *args, **kwargs):
        source, processors = self.stages[0](*args, **kwargs), self.stages[1:]
        assert isinstance(source, types.GeneratorType)
        assert all([inspect.isgeneratorfunction(processor.__call__) for processor in processors])
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), processors, source)
        yield from iter(generator)

    @property
    def stages(self): return [self.source]
    @property
    def source(self): return self.__source


class OpenPipeline(Pipeline):
    def __repr__(self): return "|".join([super().__repr__()] + [repr(processor) for processor in self.processors])
    def __init__(self, source, processors):
        assert isinstance(source, Source)
        assert isinstance(processors, list)
        assert all([isinstance(processor, Processor) for processor in processors])
        super().__init__(source)
        self.__processors = processors

    def __add__(self, stage):
        assert isinstance(stage, (Processor, Destination))
        return Pipeline(self, stage)

    @property
    def stages(self): return super().stages + list(self.processors)
    @property
    def processors(self): return self.__processors


class ClosedPipeline(OpenPipeline):
    def __repr__(self): return "|".join([super().__repr__(), repr(self.destination)])
    def __init__(self, source, processors, destination):
        assert isinstance(destination, Destination)
        super().__init__(source, processors)
        self.__destination = destination

    @property
    def stages(self): return super().stages + [self.destination]
    @property
    def destination(self): return self.__destination


class Stage(ABC):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

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
        assert inspect.isgeneratorfunction(self.execute)
        start = time.time()
        generator = self.execute(*args, **kwargs)
        for content in iter(generator):
            LOGGER.info("Source: {}|{:.2f}s".format(repr(self), time.time() - start))
            yield content
            start = time.time()


class Processor(Stage, ABC):
    def generator(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        start = time.time()
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        for content in iter(generator):
            LOGGER.info("Processor: {}|{:.2f}s".format(repr(self), time.time() - start))
            yield content
            start = time.time()


class Destination(Stage, ABC):
    def generator(self, *args, **kwargs):
        assert not inspect.isgeneratorfunction(self.execute)
        start = time.time()
        self.execute(*args, **kwargs)
        LOGGER.info("Destination: {}|{:.2f}s".format(repr(self), time.time() - start))
        return
        yield


class Downloader(Source, ABC): pass
class Calculator(Processor, ABC): pass
class Filter(Processor, ABC): pass


class Loader(Source, ABC):
    def __init__(self, *args, repository, locks, name, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        if not os.path.isdir(repository):
            raise FileNotFoundError(repository)
        self.__repository = repository
        self.__locks = locks

    @staticmethod
    def read(*args, file, filetype, **kwargs):
        content = load(*args, file=file, filetype=filetype, **kwargs)
        LOGGER.info("Load: {}".format(str(file)))
        return content

    @property
    def repository(self): return self.__repository
    @property
    def locks(self): return self.__locks


class Saver(Destination, ABC):
    def __init__(self, *args, repository, locks, name, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        if not os.path.isdir(repository):
            os.mkdir(repository)
        self.__repository = repository
        self.__locks = locks

    @staticmethod
    def write(content, *args, file, filemode, **kwargs):
        save(content, *args, file=file, mode=filemode, **kwargs)
        LOGGER.info("Save: {}".format(str(file)))

    @property
    def repository(self): return self.__repository
    @property
    def locks(self): return self.__locks


class Producer(Source, ABC):
    def __init__(self, *args, source, timeout, name, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__source = source
        self.__timeout = timeout

    def reader(self, *args, **kwargs):
        while not bool(self.source):
            try:
                yield self.source.get(timeout=self.timeout)
                self.source.done()
            except queue.Empty:
                pass
    @property
    def source(self): return self.__source
    @property
    def timeout(self): return self.__timeout


class Consumer(Destination, ABC):
    def __init__(self, *args, destination, timeout, name, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__destination = destination
        self.__timeout = timeout

    def write(self, content, *args, **kwargs):
        self.destination.put(content, timeout=self.timeout)

    @property
    def destination(self): return self.__destination
    @property
    def timeout(self): return self.__timeout







