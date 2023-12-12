# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Synchronization Objects
@author: Jack Kirby Cook

"""

import sys
import types
import queue
import inspect
import logging
import traceback
import threading

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Routine", "Producer", "Consumer", "Processor", "FIFOQueue", "LIFOQueue", "PriorityQueue"]
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
        assert isinstance(destination, Queue)
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
        assert isinstance(source, Queue)
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
        assert isinstance(source, Queue)
        assert isinstance(destination, Queue)
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


class QueueMeta(type):
    def __call__(cls, contents, *args, size=None, **kwargs):
        assert isinstance(contents, list)
        instance = super().__call__(*args, **kwargs)
        assert (len(contents) <= size) if bool(size) else True
        for content in contents:
            instance.put(content)
        return instance


class Queue(queue.Queue, metaclass=QueueMeta):
    def __repr__(self): return self.name
    def __len__(self): return self.qsize()
    def __bool__(self): return not self.empty()

    def __init__(self, *args, size=None, **kwargs):
        super().__init__(size if size is not None else 0)
        self.__obsolete = kwargs.get("obsolete", lambda content: False)
        self.__name = kwargs.get("name", self.__class__.__name__)

    def put(self, content, timeout=None):
        if not self.obsolete(content):
            super().put(content, timeout=timeout)

    def done(self): self.task_done()
    def get(self, timeout=None):
        content = super().get(timeout=timeout)
        if not self.obsolete(content):
            return content
        return self.get(content, timeout=timeout)

    @property
    def obsolete(self): return self.__obsolete
    @property
    def name(self): return self.__name


class FIFOQueue(Queue): pass
class LIFOQueue(Queue, queue.LifoQueue): pass


class PriorityQueue(Queue, queue.PriorityQueue):
    def __init__(self, *args, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(priority)
        self.__priority = priority

    def put(self, content, timeout=None):
        priority = self.priority(content)
        assert isinstance(priority, int)
        couple = (-priority, content)
        super().put(couple, timeout=timeout)

    def get(self, timeout=None):
        (priority, content) = super().get(timeout=timeout)
        return self.get(content, timeout=timeout)

    @property
    def priority(self): return self.__priority



