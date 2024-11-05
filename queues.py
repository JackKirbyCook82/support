# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from enum import StrEnum
from abc import ABC, abstractmethod

from support.mixins import Generator, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Dequeue", "Queue", "QueueTypes"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


QueueTypes = StrEnum("Typing", ["FIFO", "LIFO", "HIPO", "LIPO"], start=1)
class QueueMeta(RegistryMeta):
    def __init__(cls, *args, **kwargs):
        super(QueueMeta, cls).__init__(*args, **kwargs)
        cls.datatype = kwargs.get("datatype", getattr(cls, "datatype", None))

    def __call__(cls, *args, contents=[], capacity=None, **kwargs):
        capacity = capacity if capacity is not None else 0
        data = cls.datatype(maxsize=capacity)
        instance = super(QueueMeta, cls).__call__(*args, data=data, **kwargs)
        for content in contents:
            instance.write(content)
        return instance


class Queue(ABC, metaclass=QueueMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __bool__(self): return not bool(self.empty)
    def __len__(self): return int(self.size)

    def __init__(self, *args, data, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__timeout = timeout
        self.__data = data

    @abstractmethod
    def write(self, content, *args, **kwargs): pass
    @abstractmethod
    def read(self, *args, **kwargs): pass

    @property
    def empty(self): return self.data.empty()
    @property
    def size(self): return self.data.qsize()
    def complete(self): self.data.task_done()

    @property
    def timeout(self): return self.__timeout
    @property
    def data(self): return self.__data
    @property
    def name(self): return self.__name


class StandardQueue(Queue):
    def write(self, content, *args, **kwargs):
        self.data.put(content, timeout=self.timeout)

    def read(self, *args, **kwargs):
        content = self.data.get(timeout=self.timeout)
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
        self.data.put(couple, timeout=self.timeout)

    def read(self, *args, **kwargs):
        couple = self.data.get(timeout=self.timeout)
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


class Dequeue(Generator, Logging):
    def __init__(self, *args, **kwargs):
        Generator.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__queue = kwargs["queue"]

    def generator(self, *args, **kwargs):
        if not bool(self.queue): return
        while bool(self.queue):
            content = self.queue.read(*args, **kwargs)
            yield content
            self.queue.complete()

    @property
    def queue(self): return self.__queue




