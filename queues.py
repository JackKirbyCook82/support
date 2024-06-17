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
__all__ = ["Schedule", "Scheduler", "Queues"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Schedule(Producer):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    def execute(self, *args, **kwargs):
        while bool(self.source):
            content = self.read(*args, **kwargs)
            contents = {str(self.source.query): content}
            yield contents
            self.source.complete()

    def read(self, *args, **kwargs):
        return self.source.read(*args, **kwargs)

    @property
    def source(self): return self.__source


class Scheduler(Consumer):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.__destination = destination

    def execute(self, contents, *args, **kwargs):
        content = contents.get(str(self.destination.query), None)
        if content is not None:
            self.write(content, *args, **kwargs)

    def write(self, content, *args, **kwargs):
        self.destination.write(content, *args, **kwargs)

    @property
    def destination(self): return self.__destination


class QueueMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is QueueMeta for base in cls.__bases__]):
            return
        cls.__queuetype__ = kwargs.get("queuetype", getattr(cls, "__queuetype__", None))
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __call__(cls, *args, capacity=None, querys=[], **kwargs):
        assert cls.__queuetype__ is not None
        assert cls.__query__ is not None
        assert isinstance(querys, list)
        assert (len(querys) <= capacity) if bool(capacity) else True
        stack = cls.__queuetype__(maxsize=capacity if capacity is not None else 0)
        instance = super(QueueMeta, cls).__call__(stack, *args, **kwargs)
        for query in querys:
            assert isinstance(query, (tuple, str))
            query = instance.query(*query) if isinstance(query, tuple) else instance.query[query]
            stack.put(query)
        return instance


class Queue(ABC, metaclass=QueueMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, stack, *args, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__query = self.__class__.__query__
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
    def timeout(self): return self.__timeout
    @property
    def query(self): return self.__query
    @property
    def queue(self): return self.__queue
    @property
    def name(self): return self.__name


class StandardQueue(Queue):
    def write(self, query, *args, **kwargs):
        assert isinstance(query, self.query)
        self.queue.put(query, timeout=self.timeout)

    def read(self, *args, **kwargs):
        query = self.queue.get(timeout=self.timeout)
        assert isinstance(query, self.query)
        return query


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

    def write(self, query, *args, **kwargs):
        assert isinstance(query, self.query)
        priority = self.priority(query)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, query)
        self.queue.put(couple, timeout=self.timeout)

    def read(self, *args, **kwargs):
        couple = self.queue.get(timeout=self.timeout)
        priority, query = couple
        assert isinstance(query, self.query)
        return query

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



