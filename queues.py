# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
import logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LIFOQueue", "FIFOQueue", "LIPOQueue", "HIPOQueue"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class StandardQueue(queue.Queue):
    def __repr__(self): return self.name
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __new__(cls, contents, *args, size=None, **kwargs):
        assert isinstance(contents, list)
        assert (len(contents) <= size) if bool(size) else True
        instance = super().__new__(size if size is not None else 0)
        for content in contents:
            instance.put(content)
        return instance

    def __init__(self, *args, size=None, **kwargs):
        super().__init__(self, size if size is not None else 0)
        self.__name = kwargs.get("name", self.__class__.__name__)

    def done(self): super().task_done()
    def get(self, timeout=None): return super().get(timeout=timeout)
    def put(self, content, timeout=None): super().put(content, timeout=timeout)

    @property
    def size(self): return super().qsize()
    @property
    def empty(self): return super().empty()
    @property
    def name(self): return self.__name


class PriorityQueue(queue.PriorityQueue):
    def __repr__(self): return self.name
    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __init_subclass__(cls, *args, **kwargs):
        ascending = kwargs.get("ascending", getattr(cls, "__ascending__", True))
        assert isinstance(ascending, bool)
        cls.__ascending__ = ascending

    def __new__(cls, contents, *args, size=None, **kwargs):
        assert isinstance(contents, list)
        assert (len(contents) <= size) if bool(size) else True
        instance = super().__new__(size if size is not None else 0)
        for content in contents:
            instance.put(content)
        return instance

    def __init__(self, *args, size=None, priority, **kwargs):
        assert callable(priority)
        super().__init__(self, size if size is not None else 0)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__ascending = self.__class__.__ascending__
        self.__priority = priority

    def done(self): super().task_done()
    def get(self, timeout=None): return super().get(timeout=timeout)[1]
    def put(self, content, timeout=None):
        priority = self.priority(content)
        assert isinstance(priority, int)
        multiplier = (int(not self.ascending) * 2) - 1
        couple = (multiplier * priority, content)
        super().put(couple, timeout=timeout)

    @property
    def size(self): return super().qsize()
    @property
    def empty(self): return super().empty()

    @property
    def name(self): return self.__name
    @property
    def priority(self): return self.__priority
    @property
    def ascending(self): return self.__ascending


class LIFOQueue(StandardQueue, queue.LifoQueue): pass
class FIFOQueue(StandardQueue, queue.Queue): pass
class HIPOQueue(PriorityQueue, ascending=True): pass
class LIPOQueue(PriorityQueue, ascending=False): pass


