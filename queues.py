# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
import types
from abc import ABC

from support.pipelines import Producer, Consumer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LIFOQueue", "FIFOQueue", "LIPOQueue", "HIPOQueue", "QueueReader", "QueueWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class QueueReader(Producer, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(source, (StandardQueue, PriorityQueue))
        self.__source = source

#    def read(self, *args, **kwargs):
#        content = self.source.get()
#        self.source.done()
#        return content

#    def reader(self, *args, **kwargs):
#        while not bool(self.source):
#            try:
#                content = self.source.get()
#                self.source.done()
#                yield content
#            except queue.Empty:
#                pass

    @property
    def source(self): return self.__source
    @property
    def queue(self): return self.__source


class QueueWriter(Consumer, ABC):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(destination, (StandardQueue, PriorityQueue))
        self.__destination = destination

#    def write(self, content, *args, **kwargs):
#        self.destination.put(content)

#    def writer(self, contents, *args, **kwargs):
#        assert isinstance(contents, types.GeneratorType)
#        for content in iter(contents):
#            self.destination.put(content)

    @property
    def destination(self): return self.__destination
    @property
    def queue(self): return self.__destination


class StandardQueue(queue.Queue):
    def __bool__(self): return not self.empty
    def __repr__(self): return self.name
    def __len__(self): return self.size

    def __new__(cls, contents, *args, capacity=None, **kwargs):
        assert isinstance(contents, list)
        assert (len(contents) <= capacity) if bool(capacity) else True
        instance = super().__new__(capacity if capacity is not None else 0)
        for content in contents:
            instance.put(content)
        return instance

    def __init__(self, *args, timeout=None, capacity=None, **kwargs):
        super().__init__(self, capacity if capacity is not None else 0)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__capacity = capacity
        self.__timeout = timeout

#    def done(self): super().task_done()
#    def get(self): return super().get(timeout=self.timeout)
#    def put(self, content): super().put(content, timeout=self.timeout)

    @property
    def size(self): return super().qsize()
    @property
    def empty(self): return super().empty()

    @property
    def capacity(self): return self.__capacity
    @property
    def timeout(self): return self.__timeout
    @property
    def name(self): return self.__name


class PriorityQueue(queue.PriorityQueue):
    def __bool__(self): return not self.empty
    def __repr__(self): return self.name
    def __len__(self): return self.size

    def __init_subclass__(cls, *args, **kwargs):
        ascending = kwargs.get("ascending", getattr(cls, "__ascending__", True))
        assert isinstance(ascending, bool)
        cls.__ascending__ = ascending

    def __new__(cls, contents, *args, capacity=None, **kwargs):
        assert isinstance(contents, list)
        assert (len(contents) <= capacity) if bool(capacity) else True
        instance = super().__new__(capacity if capacity is not None else 0)
        for content in contents:
            instance.put(content)
        return instance

    def __init__(self, *args, timeout=None, capacity=None, priority, **kwargs):
        assert callable(priority)
        super().__init__(self, capacity if capacity is not None else 0)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__ascending = self.__class__.__ascending__
        self.__priority = priority
        self.__capacity = capacity
        self.__timeout = timeout

#    def done(self): super().task_done()
#    def get(self): return super().get(timeout=self.timeout)[1]
#    def put(self, content):
#        priority = self.priority(content)
#        assert isinstance(priority, int)
#        multiplier = (int(not self.ascending) * 2) - 1
#        couple = (multiplier * priority, content)
#        super().put(couple, timeout=self.timeout)

    @property
    def empty(self): return super().empty()
    @property
    def size(self): return super().qsize()

    @property
    def ascending(self): return self.__ascending
    @property
    def priority(self): return self.__priority
    @property
    def capacity(self): return self.__capacity
    @property
    def timeout(self): return self.__timeout
    @property
    def name(self): return self.__name


class LIFOQueue(StandardQueue, queue.LifoQueue): pass
class FIFOQueue(StandardQueue, queue.Queue): pass
class HIPOQueue(PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, ascending=False): pass


