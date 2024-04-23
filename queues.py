# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from abc import ABC, ABCMeta, abstractmethod

from support.pipelines import Producer, Consumer
from support.processes import Reader, Writer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Schedule", "Scheduler", "Queues"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Schedule(Reader, Producer):
    def execute(self, *args, **kwargs):
        content = self.read(*args, **kwargs)
        self.source.complete()
        contents = {self.source.variable: content}
        yield contents

    def read(self, *args, **kwargs):
        return self.source.read(*args, **kwargs)


class Scheduler(Writer, Consumer):
    def execute(self, contents, *args, **kwargs):
        content = contents.get(self.destination.variable, None)
        if content is not None:
            self.write(content, *args, **kwargs)

    def write(self, content, *args, **kwargs):
        self.destination.write(content, *args, **kwargs)


class QueueMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.Variable = kwargs.get("variable", getattr(cls, "Variable", None))
        cls.Type = kwargs.get("type", getattr(cls, "Type", None))

    def __call__(cls, *args, capacity=None, contents=[], **kwargs):
        assert cls.Variable is not None
        assert cls.Type is not None
        assert isinstance(contents, list)
        assert (len(contents) <= capacity) if bool(capacity) else True
        instance = cls.Type(maxsize=capacity if capacity is not None else 0)
        parameters = {"variable": cls.Variable}
        for content in contents:
            instance.put(content)
        wrapper = super(QueueMeta, cls).__call__(instance, *args, **parameters, **kwargs)
        return wrapper


class Queue(ABC, metaclass=QueueMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, instance, *args, variable, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__variable = variable
        self.__timeout = timeout
        self.__queue = instance

    @abstractmethod
    def write(self, content, *args, **kwargs): pass
    @abstractmethod
    def read(self, *args, **kwargs): pass

    @property
    def empty(self): return super().empty()
    @property
    def size(self): return super().qsize()
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
        ascending = self.__class__.__ascending__
        super().__init__(*args, **kwargs)
        self.__ascending = ascending
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


class FIFOQueue(StandardQueue, type=queue.Queue): pass
class LIFOQueue(StandardQueue, type=queue.LifoQueue): pass
class HIPOQueue(PriorityQueue, type=queue.PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, type=queue.PriorityQueue, ascending=False): pass


class Queues:
    FIFO = FIFOQueue
    LIFO = LIFOQueue
    HIPO = HIPOQueue
    LIPO = LIPOQueue



