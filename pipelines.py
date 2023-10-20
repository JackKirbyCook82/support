# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import os
import time
import types
import inspect
import logging
from functools import reduce
from abc import ABC, abstractmethod
from collections import OrderedDict as ODict

import support.files as files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Processor", "Calculator", "Downloader", "Uploader", "Saver", "Loader"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class Pipeline(list):
    def __repr__(self): return "|".join(list(map(repr, self)))
    def __init__(self, processors):
        assert isinstance(processors, list)
        assert all([isinstance(processor, Processor) for processor in processors])
        super().__init__(processors)

    def __add__(self, other):
        assert isinstance(other, Processor)
        return Pipeline([*self, other])

    def __call__(self, *args, **kwargs):
        source, segments = self[0](*args, **kwargs), self[1:]
        assert isinstance(source, types.GeneratorType)
        assert all([inspect.isgeneratorfunction(segment.__call__) for segment in segments])
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), segments, source)
        yield from iter(generator)


class Processor(ABC):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __add__(self, other):
        assert isinstance(other, Processor)
        return Pipeline([self, other])

    def __call__(self, *args, **kwargs):
        source = args[0] if bool(args) and isinstance(args[0], types.GeneratorType) else None
        process = self.process(source, *args, **kwargs) if source is not None else self.generator(*args, **kwargs)
        yield from iter(process)

    def process(self, source, *args, **kwargs):
        for content in iter(source):
            generator = self.generator(content, *args, **kwargs)
            yield from iter(generator)

    def generator(self, *args, **kwargs):
        start = time.time()
        if not inspect.isgeneratorfunction(self.execute):
            self.execute(*args, **kwargs)
            LOGGER.info("Processed: {}|{:.2f}s".format(repr(self), time.time() - start))
            return
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        for content in iter(generator):
            LOGGER.info("Processed: {}|{:.2f}s".format(repr(self), time.time() - start))
            yield content
            start = time.time()

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def name(self): return self.__name


class Calculator(Processor, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        calculations = {key: value for key, value in getattr(cls, "__calculations__", {}).items()}
        calculations.update({key: value for key, value in kwargs.get("calculations", {}).items()})
        cls.__calculations__ = calculations

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        calculations = ODict(list(self.__class__.__calculations__.items()))
        keys = list(kwargs.get("calculations", calculations.keys()))
        calculations = {key: value for key, value in calculations.items() if key in keys}
        calculations = [calculation(*args, **kwargs) for calculation in list(calculations.values())]
        self.__calculations = calculations

    @property
    def calculations(self): return self.__calculations


class Websites(Processor, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        pages = {key: value for key, value in getattr(cls, "__pages__", {}).items()}
        pages.update(kwargs.get("pages", {}))
        cls.__pages__ = pages

    def __getitem__(self, key): return self.pages[key]
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        pages = list(self.__class__.__pages__.items())
        pages = {key: page for key, page in iter(pages)}
        pages = {key: page(source) for key, page in pages.items()}
        self.__pages = pages

    @property
    def pages(self): return self.__pages


class Downloader(Websites, ABC): pass
class Uploader(Websites, ABC): pass


class Files(Processor, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, **kwargs)
        self.__repository = repository

    @property
    def repository(self): return self.__repository


class Loader(Files, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, repository=repository, **kwargs)
        if not os.path.isdir(repository):
            raise FileNotFoundError(repository)
        self.loader = files.Loader()
        self.reader = files.Reader()
        self.referer = files.Referer()

    def read(self, *args, file, **kwargs):
        return self.loader(*args, file=file, **kwargs)

    def reader(self, *args, file, **kwargs):
        return self.reader(*args, file=file, **kwargs)

    def refer(self, *args, file, **kwargs):
        return self.referer(*args, file=file, **kwargs)


class Saver(Files, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, repository=repository, **kwargs)
        if not os.path.isdir(repository):
            os.mkdir(repository)
        self.saver = files.Saver()

    def write(self, content, *args, file, **kwargs):
        self.saver(content, *args, file=file, **kwargs)
        LOGGER.info("Saved: {}".format(str(file)))




