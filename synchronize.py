# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Synchronization Objects
@author: Jack Kirby Cook

"""

import sys
import queue
import inspect
import logging
import traceback
import threading
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Routine", "Producer", "Consumer", "Processor", "FIFOQueue", "LIFOQueue", "HIPOQueue", "LIPOQueue", "Stack"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class Routine(threading.Thread):
    def __repr__(self): return self.name
    def __bool__(self): return super().is_alive()

    def __init__(self, routine, *args, **kwargs):
        assert callable(routine)
        name = kwargs.get("name", self.__class__.__name__)
        threading.Thread.__init__(self, name=name, daemon=False)
        self.__routine = routine
        self.__arguments = list()
        self.__parameters = dict()

    def setup(self, *args, **kwargs):
        self.arguments.extend(list(args)) if args else False
        self.parameters.update(dict(kwargs)) if kwargs else False
        return self

    def start(self, *args, **kwargs):
        LOGGER.info("Started: {}".format(repr(self)))
        threading.Thread.start(self)
        return self

    def join(self, *args, **kwargs):
        threading.Thread.join(self)
        LOGGER.info("Stopped: {}".format(repr(self)))
        return self

    def run(self):
        try:
            LOGGER.info("Running: {}".format(repr(self)))
            self.process(*self.arguments, **self.parameters)
        except BaseException as error:
            LOGGER.error("Error: {}|{}".format(repr(self), error.__class__.__name__))
            error_type, error_value, error_traceback = sys.exc_info()
            traceback.print_exception(error_type, error_value, error_traceback)
        else:
            LOGGER.info("Completed: {}".format(repr(self)))

    def process(self, *args, **kwargs):
        routine = self.routine.__call__ if hasattr(self.routine, "__call__") else self.routine
        if not inspect.isgeneratorfunction(routine):
            routine(*args, **kwargs)
        else:
            generator = routine(*args, **kwargs)
            for _ in iter(generator):
                pass

    @staticmethod
    def generator(routine):
        function = routine.__call__ if isinstance(routine, object) else routine
        if inspect.isgeneratorfunction(function):
            return function
        generator = lambda *args, **kwargs: (yield function(*args, **kwargs))
        return generator

    @property
    def routine(self): return self.__routine
    @property
    def arguments(self): return self.__arguments
    @property
    def parameters(self): return self.__parameters


class Producer(Routine):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(destination, Stack)
        self.__destination = destination

    def process(self, *args, **kwargs):
        generator = self.generator(self.routine)(*args, **kwargs)
        for content in iter(generator):
            self.destination.put(content, timeout=None)

    @property
    def destination(self): return self.__destination


class Consumer(Routine):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(source, Stack)
        self.__source = source

    def execute(self, content, *args, **kwargs): pass
    def terminate(self, *args, **kwargs): return not bool(self.source)
    def process(self, *args, **kwargs):
        while not self.terminate(*args, **kwargs):
            try:
                content = self.source.get(timeout=5)
                generator = self.generator(self.routine)(content, *args, **kwargs)
                for content in iter(generator):
                    self.execute(content, *args, **kwargs)
                self.source.done()
            except queue.Empty:
                pass

    @property
    def source(self): return self.__source


class Processor(Routine):
    def __init__(self, *args, source, destination, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(source, Stack)
        assert isinstance(destination, Stack)
        self.__source = source
        self.__destination = destination

    def terminate(self, *args, **kwargs): return not bool(self.source)
    def process(self, *args, **kwargs):
        while not self.terminate(*args, **kwargs):
            try:
                content = self.source.get(timeout=5)
                generator = self.generator(self.routine)(content, *args, **kwargs)
                for content in iter(generator):
                    self.destination.put(content, timeout=None)
                self.source.done()
            except queue.Empty:
                pass

    @property
    def source(self): return self.__source
    @property
    def destination(self): return self.__destination


class Stack(ABC):
    def __repr__(self): return self.name
    def __len__(self): return self.length
    def __bool__(self): return not self.empty

    def __init__(self, *args, limit=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__limit = limit if limit is not None else 0

    @abstractmethod
    def done(self, *args, **kwargs): pass
    @abstractmethod
    def get(self, *args, **kwargs): pass
    @abstractmethod
    def put(self, content, *args, **kwargs): pass

    @property
    @abstractmethod
    def length(self): pass
    @property
    @abstractmethod
    def empty(self): pass

    @property
    def name(self): return self.__name
    @property
    def limit(self): return self.__limit


class StandardQueue(queue.Queue, Stack):
    def __new__(cls, contents, *args, limit=None, **kwargs):
        assert isinstance(contents, list)
        assert (len(contents) <= limit) if bool(limit) else True
        instance = queue.Queue.__new__(limit if limit is not None else 0)
        for content in contents:
            instance.put(content)
        return instance

    def __init__(self, *args, limit=None, **kwargs):
        queue.Queue.__init__(self, limit if limit is not None else 0)
        Stack.__init__(self, *args, limit=limit, **kwargs)

    def done(self): super().task_done()
    def get(self, timeout=None): return super().get(timeout=timeout)
    def put(self, content, timeout=None): super().put(content, timeout=timeout)

    @property
    def length(self): return super().qsize()
    @property
    def empty(self): return super().empty()


class PriorityQueue(queue.PriorityQueue, Stack):
    def __init_subclass__(cls, *args, **kwargs):
        ascending = kwargs.get("ascending", getattr(cls, "__ascending__", True))
        assert isinstance(ascending, bool)
        cls.__ascending__ = ascending

    def __new__(cls, contents, *args, limit=None, **kwargs):
        assert isinstance(contents, list)
        assert (len(contents) <= limit) if bool(limit) else True
        instance = queue.PriorityQueue.__new__(limit if limit is not None else 0)
        for content in contents:
            instance.put(content)
        return instance

    def __init__(self, *args, limit=None, priority, **kwargs):
        queue.PriorityQueue.__init__(self, limit if limit is not None else 0)
        Stack.__init__(self, *args, limit=limit, **kwargs)
        assert callable(priority)
        self.__ascending = self.__class__.__ascending__
        self.__priority = priority

    def done(self): super().task_done()
    def get(self, timeout=None): return super().get(timeout=timeout)[1]
    def put(self, content, timeout=None):
        priority = self.priority(content)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, content)
        super().put(couple, timeout=timeout)

    @property
    def length(self): return super().qsize()
    @property
    def empty(self): return super().empty()

    @property
    def priority(self): return self.__priority
    @property
    def ascending(self): return self.__ascending


class LIFOQueue(queue.LifoQueue, StandardQueue): pass
class FIFOQueue(queue.Queue, StandardQueue): pass
class HIPOQueue(PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, ascending=False): pass


