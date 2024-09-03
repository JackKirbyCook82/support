# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from abc import ABC, abstractmethod

from support.pipelines import Producer, Consumer
from support.mixins import Mixin

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Dequeuer", "Requeuer", "Queues"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class QueueMixin(Mixin):
    def __init__(self, *args, datastack, **kwargs):
        assert isinstance(datastack, Queue)
        super().__init__(*args, **kwargs)
        self.__datastack = datastack

    @property
    def datastack(self): return self.__datastack


class Dequeuer(QueueMixin, Producer, title="Dequeued"):
    def producer(self, *args, **kwargs):
        while bool(self.datastack):
            variable = self.read(*args, **kwargs)
            contents = {self.variable: variable}
            yield contents
            self.datastack.complete()

    def read(self, *args, **kwargs):
        return self.datastack.read(*args, **kwargs)


class Requeuer(QueueMixin, Consumer, title="Requeued"):
    def consumer(self, contents, *args, **kwargs):
        variable = contents[self.variable]
        self.write(variable, *args, **kwargs)

    def write(self, value, *args, **kwargs):
        self.datastack.write(value, *args, **kwargs)


class Queue(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__queue__ = kwargs.get("queue", getattr(cls, "__queue__", None))

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __init__(self, *args, contents=[], capacity=None, timeout=None, **kwargs):
        assert isinstance(contents, list) and (len(contents) <= capacity if bool(capacity) else True)
        capacity = capacity if capacity is not None else 0
        container = self.__class__.__queue__
        assert container is not None
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__queue = container(maxsize=capacity)
        self.__timeout = timeout
        for content in contents:
            self.put(content)

    @abstractmethod
    def write(self, value, *args, **kwargs): pass
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
    def write(self, value, *args, **kwargs):
        self.queue.put(value, timeout=self.timeout)

    def read(self, *args, **kwargs):
        value = self.queue.get(timeout=self.timeout)
        return value


class PriorityQueue(Queue):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        ascending = kwargs.get("ascending", getattr(cls, "__ascending__", True))
        assert isinstance(ascending, bool)
        cls.__ascending__ = ascending

    def __init__(self, *args, priority, **kwargs):
        assert callable(priority)
        super().__init__(*args, **kwargs)
        self.__ascending = self.__class__.__ascending__
        self.__priority = priority

    def write(self, value, *args, **kwargs):
        priority = self.priority(value)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, value)
        self.queue.put(couple, timeout=self.timeout)

    def read(self, *args, **kwargs):
        couple = self.queue.get(timeout=self.timeout)
        priority, value = couple
        return value

    @property
    def ascending(self): return self.__ascending
    @property
    def priority(self): return self.__priority


class FIFOQueue(StandardQueue, queue=queue.Queue): pass
class LIFOQueue(StandardQueue, queue=queue.LifoQueue): pass
class HIPOQueue(PriorityQueue, queue=queue.PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, queue=queue.PriorityQueue, ascending=False): pass


class Queues(object):
    FIFO = FIFOQueue
    LIFO = LIFOQueue
    HIPO = HIPOQueue
    LIPO = LIPOQueue



