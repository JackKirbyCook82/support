# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from abc import ABC

from support.pipelines import Stack

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LIFOQueue", "FIFOQueue", "LIPOQueue", "HIPOQueue"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class Queue(Stack, ABC):
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __init__(self, contents, *args, capacity=None, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(contents, list)
        assert (len(contents) <= capacity) if bool(capacity) else True
        self.__queue = self.type(capacity if capacity is not None else 0)
        self.__timeout = timeout
        for content in contents:
            self.write(content)

    @property
    def empty(self): return super().empty()
    @property
    def size(self): return super().qsize()

    @property
    def timeout(self): return self.__timeout
    @property
    def queue(self): return self.__queue


class StandardQueue(Queue):
    def read(self, *args, **kwargs):
        content = self.queue.get(timeout=self.timeout)
        self.queue.task_done()
        return content

    def write(self, content, *args, **kwargs):
        self.queue.put(content, timeout=self.timeout)


class PriorityQueue(Queue):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        ascending = kwargs.get("ascending", getattr(cls, "__ascending__", True))
        assert isinstance(ascending, bool)
        cls.__ascending__ = ascending

    def __init__(self, contents, *args, priority, **kwargs):
        assert callable(priority)
        ascending = self.__class__.__ascending__
        super().__init__(*args, **kwargs)
        self.__ascending = ascending
        self.__priority = priority
        for content in contents:
            self.write(content)

    def read(self, *args, **kwargs):
        couple = self.queue.get(timeout=self.timeout)
        priority, content = couple
        self.queue.task_done()
        return content

    def write(self, content, *args, **kwargs):
        priority = self.priority(content)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, content)
        self.queue.put(couple, timeout=self.timeout)

    @property
    def ascending(self): return self.__ascending
    @property
    def priority(self): return self.__priority


class FIFOQueue(StandardQueue, type=queue.Queue): pass
class LIFOQueue(StandardQueue, type=queue.LifoQueue): pass
class HIPOQueue(PriorityQueue, type=queue.PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, type=queue.PriorityQueue, ascending=False): pass


