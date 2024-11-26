# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from enum import StrEnum
from abc import ABC, ABCMeta, abstractmethod

from support.meta import RegistryMeta
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Dequeuer", "Queue", "QueueTypes"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


QueueTypes = StrEnum("Typing", ["FIFO", "LIFO", "PIFO"], start=1)
class QueueMeta(RegistryMeta, ABCMeta):
    def __init__(cls, *args, **kwargs):
        register = kwargs.get("queuetype", [])
        super(QueueMeta, cls).__init__(*args, register=register, **kwargs)
        cls.__queuetype__ = kwargs.get("queuetype", getattr(cls, "__queuetype__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))

    def __call__(cls, *args, contents=[], capacity=None, **kwargs):
        cls = cls[kwargs["queuetype"]] if cls.queuetype is None else cls
        capacity = capacity if capacity is not None else 0
        data = cls.datatype(maxsize=capacity)
        instance = super(QueueMeta, cls).__call__(*args, data=data, **kwargs)
        for content in contents: instance.write(content)
        return instance

    @property
    def queuetype(cls): return cls.__queuetype__
    @property
    def datatype(cls): return cls.__datatype__


class Queue(Logging, ABC, metaclass=QueueMeta):
    def __repr__(self): return f"{str(self.name)}[{len(self):.0f}]"
    def __bool__(self): return not bool(self.empty)
    def __len__(self): return int(self.size)

    def __init__(self, *args, data, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
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


class FIFOQueue(StandardQueue, datatype=queue.Queue, queuetype=QueueTypes.FIFO): pass
class LIFOQueue(StandardQueue, datatype=queue.LifoQueue, queuetype=QueueTypes.LIFO): pass
class PIFOQueue(Queue, datatype=queue.PriorityQueue, queuetype=QueueTypes.PIFO):
    def __init__(self, *args, priority, ascending, **kwargs):
        assert callable(priority) and isinstance(ascending, bool)
        super().__init__(*args, **kwargs)
        self.__ascending = ascending
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


class Dequeuer(Logging):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__queue = kwargs["queue"]

    def execute(self, *args, **kwargs):
        if not bool(self.queue): return
        while bool(self.queue):
            content = self.queue.read(*args, **kwargs)
            yield content
            self.queue.complete()

    @property
    def queue(self): return self.__queue




