# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from enum import StrEnum
from abc import ABC, abstractmethod

from support.mixins import Function, Generator, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Dequeue", "Requeue", "Queue", "QueueTypes"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


QueueTypes = StrEnum("Typing", ["FIFO", "LIFO", "HIPO", "LIPO"], start=1)
class QueueMeta(RegistryMeta):
    def __init__(cls, *args, **kwargs):
        super(QueueMeta, cls).__init__(*args, **kwargs)
        cls.datatype = kwargs.get("datatype", getattr(cls, "datatype", None))

    def __call__(cls, *args, **kwargs):
        parameters = dict(name=)
        instance = super(QueueMeta, cls).__call__(*args, **kwargs)
        return instance


class Queue(ABC, metaclass=QueueMeta):
    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __bool__(self): return not bool(self.empty)
    def __len__(self): return int(self.size)

    def __init__(self, *args, contents=[], capacity=None, timeout=None, **kwargs):
        capacity = capacity if capacity is not None else 0
        self.__queue = self.__class__.__datatype__(maxsize=capacity)
        self.__name = self.__class__.__name__
        self.__timeout = timeout
        for content in contents:
            self.put(content)

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
    def timeout(self): return self.__timeout
    @property
    def queue(self): return self.__queue
    @property
    def name(self): return self.__name


class StandardQueue(Queue):
    def write(self, content, *args, **kwargs):
        self.queue.put(content, timeout=self.timeout)

    def read(self, *args, **kwargs):
        content = self.queue.get(timeout=self.timeout)
        return content


class PriorityQueue(Queue, datatype=queue.PriorityQueue):
    def __init_subclass__(cls, *args, **kwargs):
        Queue.__init_subclass__(cls, *args, **kwargs)
        ascending = kwargs.get("ascending", getattr(cls, "__ascending__", True))
        assert isinstance(ascending, bool)
        cls.__ascending__ = ascending

    def __init__(self, *args, priority, **kwargs):
        assert callable(priority)
        Queue.__init__(self, *args, **kwargs)
        self.__ascending = self.__class__.__ascending__
        self.__priority = priority

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


class FIFOQueue(StandardQueue, datatype=queue.Queue, register=QueueTypes.FIFO): pass
class LIFOQueue(StandardQueue, datatype=queue.LifoQueue, register=QueueTypes.LIFO): pass
class HIPOQueue(PriorityQueue, ascending=True, register=QueueTypes.HIPO): pass
class LIPOQueue(PriorityQueue, ascending=False, register=QueueTypes.LIPO): pass


class Requeue(Function, Logging, ABC):
    pass


class Dequeue(Generator, Logging, ABC):
    pass




