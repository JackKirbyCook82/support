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
    def __init__(self, *args, **kwargs):
        assert isinstance(kwargs["queue"], Queue)
        super().__init__(*args, **kwargs)
        self.__queue = kwargs["queue"]

    def read(self, *args, **kwargs): return self.queue.read(*args, **kwargs)
    def producer(self, *args, **kwargs):
        while bool(self.queue):
            variable = self.read(*args, **kwargs)
            contents = {self.variable: variable}
            yield contents
            self.datastack.complete()

    @property
    def queue(self): return self.__queue


class Requeuer(Consumer, title="Requeued"):
    def __init__(self, *args, datastack, **kwargs):
        assert isinstance(kwargs["queue"], Queue)
        super().__init__(*args, **kwargs)
        self.__queue = kwargs["queue"]

    def write(self, value, *args, **kwargs): self.queue.write(value, *args, **kwargs)
    def consumer(self, contents, *args, **kwargs):
        variable = contents[self.variable]
        self.write(variable, *args, **kwargs)

    @property
    def queue(self): return self.__queue


class QueueMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is QueueMeta for base in cls.__bases__]):
            return
        cls.__queuetype__ = kwargs.get("queuetype", getattr(cls, "__queuetype__", None))

    def __call__(cls, *args, contents=[], capacity=None, **kwargs):
        assert isinstance(contents, list) and (len(contents) <= capacity if bool(capacity) else True)
        capacity = capacity if capacity is not None else 0
        wrapped = cls.__queuetype__(maxsize=capacity)
        instance = super(QueueMeta, cls).__call__(*args, queue=wrapped, **kwargs)
        for content in contents:
            instance.put(content)
        return instance


class Queue(ABC, metaclass=QueueMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __init__(self, *args, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__queuedata = kwargs["queue"]
        self.__timeout = timeout

    @abstractmethod
    def write(self, value, *args, **kwargs): pass
    @abstractmethod
    def read(self, *args, **kwargs): pass

    @property
    def empty(self): return self.queuedata.empty()
    @property
    def size(self): return self.queuedata.qsize()
    def complete(self): self.queuedata.task_done()

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


class FIFOQueue(StandardQueue, queuetype=queue.Queue): pass
class LIFOQueue(StandardQueue, queuetype=queue.LifoQueue): pass
class HIPOQueue(PriorityQueue, queuetype=queue.PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, queuetype=queue.PriorityQueue, ascending=False): pass


class Queues(object):
    FIFO = FIFOQueue
    LIFO = LIFOQueue
    HIPO = HIPOQueue
    LIPO = LIPOQueue



