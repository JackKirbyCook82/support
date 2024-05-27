# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import inspect
import logging
from abc import ABC, abstractmethod
from functools import update_wrapper

from support.meta import AttributeMeta
from support.mixins import Fields

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Query", "Header", "Stage", "Routine", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class HeaderMeta(AttributeMeta): pass
class Header(Fields, metaclass=HeaderMeta): pass
class DataFrameHeader(Header, fields=["index", "columns"], register="Dataframe"):
    def __call__(self, dataframe):
        if not set(self.index) == set(dataframe.index.values):
            index = [column for column in self.index if column in dataframe.columns]
            dataframe = dataframe.set_index(index, drop=True, inplace=False)
        if not set(self.columns) == set(dataframe.columns):
            columns = [column for column in self.columns if column in dataframe.columns]
            dataframe = dataframe[columns]
        return dataframe


class Query(object):
    def __new__(cls, arguments=[], parameters={}, headers={}):
        assert isinstance(arguments, list) and isinstance(parameters, dict)
        assert all([isinstance(parameter, list) for parameter in parameters.values()])

        arguments = lambda query: {argument: query.get(argument, None) for argument in arguments}
        parameters = lambda query: {parameter: {content: query.get(content, None) for content in contents} for parameter, contents in parameters.items()}

        def extract(query):
            assert isinstance(query, dict)
            return arguments(query) | parameters(query)

        def parse(results):
            assert isinstance(results, dict)
            updated = {key: value for key, value in results.items() if key in headers.keys()}
            updated = {key: headers[key](value) for key, value in updated.items()}
            return results | updated

        def decorator(execute):
            assert inspect.isgeneratorfunction(execute)

            def wrapper(self, query, *args, **kwargs):
                feed = extract(query)
                for results in execute(self, *args, **feed, **kwargs):
                    results = parse(results)
                    yield query | results

            update_wrapper(wrapper, execute)
            return wrapper
        return decorator


class Stage(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, name=None, title=None, **kwargs):
        self.__title = title if title is not None else self.__class__.__title__
        self.__name = name if name is not None else self.__class__.__name__

    def logger(self, elapsed):
        __logger__.info(f"{self.title}: {repr(self)}[{elapsed:.2f}s]")

    @abstractmethod
    def process(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name
    @property
    def title(self): return self.__title


class Routine(Stage, ABC, title="Performed"):
    def __call__(self, *args, **kwargs):
        self.process(*args, **kwargs)

    def process(self, *args, **kwargs):
        start = time.time()
        self.execute(*args, **kwargs)
        self.logger(time.time() - start)


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
        for content in iter(generator):
            self.logger(time.time() - start)
            yield content
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
            start = time.time()
            self.execute(query, *args, **kwargs)
            self.logger(time.time() - start)




