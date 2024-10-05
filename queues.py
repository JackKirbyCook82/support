# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from abc import ABC, abstractmethod

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LIFOQueue", "FIFOQueue", "LIPOQueue", "HIPOQueue"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Queue(Logging, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__type__ = kwargs.get("queuetype", getattr(cls, "__queuetype__", None))

    def __repr__(self): return f"{self.name}[{len(self):.0f}]"
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __init__(self, *args, contents=[], capacity=None, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        capacity = capacity if capacity is not None else 0
        self.__queue = self.__class__.__type__(maxsize=capacity)
        self.__timeout = timeout
        for content in contents:
            self.put(content)

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


class StandardQueue(Queue):
    def write(self, content, *args, **kwargs):
        self.queue.put(content, timeout=self.timeout)

    def read(self, *args, **kwargs):
        content = self.queue.get(timeout=self.timeout)
        return content


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

    def write(self, content, *args, **kwargs):
        priority = self.priority(content)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, content)
        self.queue.put(couple, timeout=self.timeout)

    def read(self, *args, **kwargs):
        couple = self.queue.get(timeout=self.timeout)
        priority, content = couple
        return content

    @property
    def ascending(self): return self.__ascending
    @property
    def priority(self): return self.__priority


class FIFOQueue(StandardQueue, type=queue.Queue): pass
class LIFOQueue(StandardQueue, type=queue.LifoQueue): pass
class HIPOQueue(PriorityQueue, type=queue.PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, type=queue.PriorityQueue, ascending=False): pass





