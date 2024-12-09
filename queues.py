# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from enum import StrEnum
from abc import ABC, ABCMeta, abstractmethod

from support.meta import AttributeMeta
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Dequeuer", "Queue"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


QueueTypes = StrEnum("QueueTypes", "LIFO FIFO PIFO")
class QueueMeta(AttributeMeta, ABCMeta):
    def __init__(cls, name, bases, attrs, *args, queuetype=None, datatype=None, **kwargs):
        if not any([type(base) is QueueMeta for base in bases]):
            super(QueueMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
            return
        if ABC in bases:
            return
        assert all([queuetype is not None, datatype is not None])
        super(QueueMeta, cls).__init__(name, bases, attrs, *args, attribute=str(queuetype.name), **kwargs)
        cls.__queuetype__ = queuetype
        cls.__datatype__ = datatype

    def __call__(cls, *args, contents=[], capacity=None, **kwargs):
        assert all([cls.queuetype is not None, cls.datatype is not None])
        data = cls.datatype(maxsize=capacity if capacity is not None else 0)
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


class StandardQueue(Queue, ABC):
    def write(self, content, *args, **kwargs):
        self.data.put(content, timeout=self.timeout)

    def read(self, *args, **kwargs):
        content = self.data.get(timeout=self.timeout)
        return content


class LIFOQueue(StandardQueue, datatype=queue.LifoQueue, queuetype=QueueTypes.LIFO): pass
class FIFOQueue(StandardQueue, datatype=queue.Queue, queuetype=QueueTypes.FIFO): pass
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




