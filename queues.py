# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from abc import ABC, abstractmethod

from support.pipelines import Producer, Consumer
from support.meta import AttributeMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Schedule", "Scheduler", "Queue"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Schedule(Producer):
    def __init__(self, *args, source, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__source = source

    def execute(self, *args, **kwargs):
        while bool(self.source):
            content = self.read(*args, **kwargs)
            contents = {self.source.variable: content}
            yield contents
            self.source.complete()

    def read(self, *args, **kwargs):
        return self.source.read(*args, **kwargs)

    @property
    def source(self): return self.__source


class Scheduler(Consumer):
    def __init__(self, *args, destination, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__destination = destination

    def execute(self, contents, *args, **kwargs):
        content = contents.get(self.destination.variable, None)
        if content is not None:
            self.write(content, *args, **kwargs)

    def write(self, content, *args, **kwargs):
        self.destination.write(content, *args, **kwargs)

    @property
    def destination(self): return self.__destination


class QueueMeta(AttributeMeta):
    def __init__(cls, *args, **kwargs):
        super(QueueMeta, cls).__init__(*args, **kwargs)
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__type__ = kwargs.get("type", getattr(cls, "__type__", None))

    def __call__(cls, *args, capacity=None, contents=[], **kwargs):
        assert cls.__variable__ is not None
        assert cls.__type__ is not None
        assert isinstance(contents, list)
        assert (len(contents) <= capacity) if bool(capacity) else True
        parameters = dict(maxsize=capacity if capacity is not None else 0)
        stack = cls.__type__(**parameters)
        for content in contents:
            stack.put(content)
        instance = super(QueueMeta, cls).__call__(stack, *args, **kwargs)
        return instance


class Queue(ABC, metaclass=QueueMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, stack, *args, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__variable = self.__class__.__variable__
        self.__timeout = timeout
        self.__queue = stack

    @abstractmethod
    def write(self, content, *args, **kwargs): pass
    @abstractmethod
    def read(self, *args, **kwargs): pass

    @property
    def empty(self): return self.queue.empty()
    @property
    def size(self): return self.queue.qsize()
    def complete(self): self.queue.task_done()

    @property
    def queue(self): return self.__queue
    @property
    def variable(self): return self.__variable
    @property
    def timeout(self): return self.__timeout
    @property
    def name(self): return self.__name


class StandardQueue(Queue):
    def write(self, content, *args, **kwargs):
        self.queue.put(content, timeout=self.timeout)

    def read(self, *args, **kwargs):
        content = self.queue.get(timeout=self.timeout)
        return content


class PriorityQueue(Queue):
    def __init_subclass__(cls, *args, **kwargs):
        ascending = kwargs.get("ascending", getattr(cls, "__ascending__", True))
        assert isinstance(ascending, bool)
        cls.__ascending__ = ascending

    def __init__(self, contents, *args, priority, **kwargs):
        assert callable(priority)
        super().__init__(*args, **kwargs)
        self.__ascending = self.__class__.__ascending__
        self.__priority = priority
        for content in contents:
            self.write(content)

    def write(self, content, *args, **kwargs):
        priority = self.priority(content)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, content)
        self.queue.put(couple, timeout=self.timeout)

    def read(self, *args, **kwargs):
        couple = self.queue.get(timeout=self.timeout)
        priority, content = couple
        return content

    @property
    def ascending(self): return self.__ascending
    @property
    def priority(self): return self.__priority


class FIFOQueue(StandardQueue, type=queue.Queue, attribute="FIFO"): pass
class LIFOQueue(StandardQueue, type=queue.LifoQueue, attribute="LIFO"): pass
class HIPOQueue(PriorityQueue, type=queue.PriorityQueue, attribute="HIPO", ascending=True): pass
class LIPOQueue(PriorityQueue, type=queue.PriorityQueue, attribute="LIPO", ascending=False): pass



