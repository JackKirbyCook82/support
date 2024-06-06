# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import time
import types
import logging
from abc import ABC, ABCMeta, abstractmethod

from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Stage", "Routine", "Producer", "Processor", "Consumer", "Header"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)

import pandas as pd


class Stage(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, name=None, title=None, **kwargs):
        self.__title = title if title is not None else self.__class__.__title__
        self.__name = name if name is not None else self.__class__.__name__

    def logger(self, *args, elapsed, **kwargs):
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
        elapsed = time.time() - start
        self.logger(elapsed=elapsed)


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
            elapsed = time.time() - start
            self.logger(elapsed=elapsed)
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
            elapsed = time.time() - start
            self.logger(elapsed=elapsed)


class AxesMeta(RegistryMeta):
    def __init__(cls, *args, datatype=None, **kwargs):
        super(AxesMeta, cls).__init__(*args, key=datatype, **kwargs)


class Axes(ABC, metaclass=AxesMeta):
    @abstractmethod
    def parse(self, content, *args, **kwargs): pass


class Dataframe(Axes, datatype=pd.DataFrame):
    def __init__(self, *args, index, columns, duplicates=True, **kwargs):
        assert not set(index) & set(columns)
        self.__duplicates = duplicates
        self.__columns = columns
        self.__index = index

    def parse(self, dataframe, *args, **kwargs):
        index = [value for value in self.index if value in dataframe.columns]
        columns = [value for value in self.columns if value in dataframe.columns]
        dataframe = dataframe.drop_duplicates(index, inplace=False) if not self.duplicates else dataframe
        dataframe = dataframe.set_index(index, drop=True, inplace=False)
        dataframe = dataframe[columns]
        return dataframe

    @property
    def duplicates(self): return self.__duplicates
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class HeaderMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is HeaderMeta for base in cls.__bases__]):
            return
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__axes__ = kwargs.get("axes", getattr(cls, "__axes__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__variable__ is not None
        assert cls.__datatype__ is not None
        assert cls.__axes__ is not None
        datatype, axes = cls.__datatype__, cls.__axes__
        axes = Axes[datatype](*args, **axes, **kwargs)
        parameters = dict(axes=axes)
        instance = super(HeaderMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Header(Processor, metaclass=HeaderMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, axes, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variable = self.__class__.__variable__
        self.__axes = axes

    def execute(self, contents, *args, **kwargs):
        contents = self.axes.parse(contents, *args, **kwargs)
        return contents

    @property
    def duplicates(self): return self.__duplicates
    @property
    def variable(self): return self.__variable
    @property
    def axes(self): return self.__axes



