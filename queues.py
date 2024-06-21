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


class Queuer(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __new__(cls, *args, **kwargs):
        assert cls.__query__ is not None
        assert isinstance(cls.__query__, tuple)
        assert len(cls.__query__) == 2
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        queryname, querytype = self.__class__.__query__
        self.__queryname = queryname
        self.__querytype = querytype

    @property
    def queryname(self): return self.__queryname
    @property
    def querytype(self): return self.__querytype


class Dequeuer(Queuer, Producer, title="Dequeued"):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    def execute(self, *args, **kwargs):
        while bool(self.source):
            query = self.read(*args, **kwargs)
            assert isinstance(query, self.querytype)
            contents = {str(self.queryname): query}
            yield contents
            self.source.complete()

    def read(self, *args, **kwargs):
        return self.source.read(*args, **kwargs)

    @property
    def source(self): return self.__source


class Requeuer(Queuer, Consumer, title="Requeued"):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.__destination = destination

    def execute(self, contents, *args, **kwargs):
        query = contents[str(self.queryname)]
        assert isinstance(query, self.querytype)
        self.write(query, *args, **kwargs)

    def write(self, content, *args, **kwargs):
        self.destination.write(content, *args, **kwargs)

    @property
    def destination(self): return self.__destination


class QueueMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is QueueMeta for base in cls.__bases__]):
            return
        cls.__queuetype__ = kwargs.get("queuetype", getattr(cls, "__queuetype__", None))

    def __call__(cls, *args, capacity=None, contents=[], **kwargs):
        assert cls.__queuetype__ is not None
        assert isinstance(contents, list)
        assert (len(contents) <= capacity) if bool(capacity) else True
        queuetype = cls.__queuetype__
        instance = queuetype(maxsize=capacity if capacity is not None else 0)
        for content in contents:
            instance.put(content)
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
    def write(self, query, *args, **kwargs):
        self.queue.put(query, timeout=self.timeout)

    def read(self, *args, **kwargs):
        query = self.queue.get(timeout=self.timeout)
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
        priority = self.priority(query)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, query)
        self.queue.put(couple, timeout=self.timeout)

    def read(self, *args, **kwargs):
        couple = self.queue.get(timeout=self.timeout)
        priority, query = couple
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



