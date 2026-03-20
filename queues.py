# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import types
import queue
from abc import ABC, ABCMeta, abstractmethod

from support.concepts import Assembly

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Queues"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class QueueMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        super(QueueMeta, cls).__init__(*args, **kwargs)
        cls.__queuetype__ = kwargs.get("queuetype", getattr(cls, "__queuetype__", None))

    def __call__(cls, *args, contents, capacity=None, **kwargs):
        assert cls.queuetype is not None
        assert isinstance(contents, (list, types.NoneType))
        queuedata = cls.queuetype(maxsize=capacity if capacity is not None else 0)
        instance = super(QueueMeta, cls).__call__(*args, queuedata=queuedata, **kwargs)
        for content in contents: instance.write(content)
        return instance

    @property
    def queuetype(cls): return cls.__queuetype__


class Queue(ABC, metaclass=QueueMeta):
    def __bool__(self): return not bool(self.empty)
    def __len__(self): return int(self.size)

    def __init__(self, *args, queuedata, timeout=None, **kwargs):
        self.__queuedata = queuedata
        self.__timeout = timeout

    @abstractmethod
    def write(self, content): pass
    @abstractmethod
    def read(self): pass

    @property
    def empty(self): return self.queuedata.empty()
    @property
    def size(self): return self.queuedata.qsize()
    def complete(self): self.queuedata.task_done()

    @property
    def queuedata(self): return self.__queuedata
    @property
    def timeout(self): return self.__timeout


class StandardQueue(Queue, ABC):
    def write(self, content):
        self.queuedata.put(content, timeout=self.timeout)

    def read(self):
        content = self.queuedata.get(timeout=self.timeout)
        return content


class LIFOQueue(StandardQueue, queuetype=queue.LifoQueue): pass
class FIFOQueue(StandardQueue, queuetype=queue.Queue): pass
class PIFOQueue(Queue, queuetype=queue.PriorityQueue):
    def __init__(self, *args, priority, ascending, **kwargs):
        assert callable(priority) and isinstance(ascending, bool)
        super().__init__(*args, **kwargs)
        self.__ascending = ascending
        self.__priority = priority

    def write(self, content):
        priority = self.priority(content)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        namespace = types.SimpleNamespace(content=content, priority=priority * multiplier)
        self.queuedata.put(namespace, timeout=self.timeout)

    def read(self):
        namespace = self.queuedata.get(timeout=self.timeout)
        return namespace.content

    @property
    def ascending(self): return self.__ascending
    @property
    def priority(self): return self.__priority


class Queues(Assembly):
    LIFO = LIFOQueue
    FIFO = FIFOQueue
    PIFO = PIFOQueue

