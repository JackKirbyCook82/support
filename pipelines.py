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
from abc import ABC, abstractmethod

from support.meta import Meta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Routine", "CycleProducer", "Producer", "Processor", "Consumer", "Breaker"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Breaker(object):
    pass


class Stage(ABC, metaclass=Meta):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__title = kwargs.get("title", self.__class__.__title__)
        self.__name = kwargs.get("name", self.__class__.__name__)

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
        assert not inspect.isgeneratorfunction(self.process)
        self.process(*args, **kwargs)

    def process(self, *args, **kwargs):
        assert not inspect.isgeneratorfunction(self.execute)
        start = time.time()
        self.execute(*args, **kwargs)
        __logger__.info(f"{self.title}: {repr(self)}[{time.time() - start:.2f}s]")


class Generator(Stage, ABC, title="Generated"):
    def __repr__(self):
        terminal = bool(self.stage is None)
        strings = [super().__repr__(), repr(self.stage) if not terminal else None]
        return "|".join(list(filter(strings)))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__stage = None

    def __add__(self, stage):
        assert isinstance(stage, (Processor, Consumer))
        terminal = bool(self.stage is None)
        self.stage = stage if terminal else (self.stage + stage)
        return self

    def __call__(self, *args, **kwargs):
        terminal = bool(self.stage is None)
        assert inspect.isgeneratorfunction(self.process)
        generator = self.process(*args, **kwargs)
        if terminal:
            yield from generator
        elif isinstance(self.stage, Generator):
            generator = self.stage(generator, *args, **kwargs)
            yield from generator
        else:
            self.stage(generator, *args, **kwargs)
            return

    def generator(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        generator = self.execute(*args, **kwargs)
        start = time.time()
        for content in iter(generator):
            __logger__.info(f"{self.title}: {repr(self)}[{time.time() - start:.2f}s]")
            yield content
            start = time.time()

    @property
    def stage(self): return self.__stage
    @stage.setter
    def stage(self, stage): self.__stage = stage


class Producer(Generator, ABC, title="Produced"):
    def process(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.generator)
        generator = self.generator(*args, **kwargs)
        yield from generator


class CycleProducer(Producer, ABC):
    def __init__(self, *args, breaker, wait=None, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__breaker = breaker
        self.__wait = wait

    @staticmethod
    def prepare(*args, **kwargs): return {}
    def process(self, *args, **kwargs):
        parameters = self.prepare(*args, **kwargs)
        kwargs = kwargs | parameters
        while bool(self.breaker):
            assert inspect.isgeneratorfunction(self.generator)
            generator = self.generator(*args, **kwargs)
            yield from generator
            time.sleep(self.wait) if self.wait is not None else True

    @property
    def breaker(self): return self.__breaker
    @property
    def wait(self): return self.__wait


class Processor(Generator, ABC, title="Processed"):
    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for query in iter(stage):
            assert inspect.isgeneratorfunction(self.generator)
            generator = self.generator(query, *args, **kwargs)
            yield from generator


class Consumer(Stage, ABC, title="Consumed"):
    def __call__(self, *args, **kwargs):
        assert not inspect.isgeneratorfunction(self.process)
        self.process(*args, **kwargs)

    def process(self, stage, *args, **kwargs):
        assert isinstance(stage, types.GeneratorType)
        for query in iter(stage):
            start = time.time()
            assert not inspect.isgeneratorfunction(self.execute)
            self.execute(query, *args, **kwargs)
            __logger__.info(f"{self.title}: {repr(self)}[{time.time() - start:.2f}s]")




