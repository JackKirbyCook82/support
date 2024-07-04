# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from abc import ABC, ABCMeta, abstractmethod

from support.pipelines import Producer, Consumer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Dequeuer", "Requeuer", "Queues"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Dequeuer(Producer, title="Dequeued"):
    def __init_subclass__(cls, *args, query, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = query

    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__query = self.__class__.__query__
        self.__source = source

    def execute(self, *args, **kwargs):
        while bool(self.source):
            value = self.read(*args, **kwargs)
            values = {self.query: value}
            yield values
            self.source.complete()

    def read(self, *args, **kwargs):
        return self.source.read(*args, **kwargs)

    @property
    def source(self): return self.__source
    @property
    def query(self): return self.__query


class Requeuer(Consumer, title="Requeued"):
    def __init_subclass__(cls, *args, query, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = query

    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.__query = self.__class__.__query__
        self.__destination = destination

    def execute(self, values, *args, **kwargs):
        value = values[self.query]
        self.write(value, *args, **kwargs)

    def write(self, value, *args, **kwargs):
        self.destination.write(value, *args, **kwargs)

    @property
    def destination(self): return self.__destination
    @property
    def query(self): return self.__query


class QueueMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is QueueMeta for base in cls.__bases__]):
            return
        cls.__queuetype__ = kwargs.get("queuetype", getattr(cls, "__queuetype__", None))

    def __call__(cls, *args, capacity=None, values=[], **kwargs):
        assert cls.__queuetype__ is not None
        assert isinstance(values, list)
        assert (len(values) <= capacity) if bool(capacity) else True
        instance = cls.__queuetype__(maxsize=capacity if capacity is not None else 0)
        for value in values:
            instance.put(value)
        instance = super(QueueMeta, cls).__call__(instance, *args, **kwargs)
        return instance


class Queue(ABC, metaclass=QueueMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, instance, *args, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__timeout = timeout
        self.__queue = instance

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


class FIFOQueue(StandardQueue, queuetype=queue.Queue): pass
class LIFOQueue(StandardQueue, queuetype=queue.LifoQueue): pass
class HIPOQueue(PriorityQueue, queuetype=queue.PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, queuetype=queue.PriorityQueue, ascending=False): pass


class Queues(object):
    FIFO = FIFOQueue
    LIFO = LIFOQueue
    HIPO = HIPOQueue
    LIPO = LIPOQueue



