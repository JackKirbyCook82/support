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
__all__ = ["Query", "Header", "Stage", "Routine", "Producer", "Processor", "Consumer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Header:
    pass


class Query:
    pass


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




