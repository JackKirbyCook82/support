# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Queue Objects
@author: Jack Kirby Cook

"""

import queue
from abc import ABC, ABCMeta, abstractmethod

from support.pipelines import Producer, Consumer
from support.processes import Writer, Reader

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Populate", "Depopulate", "Stack", "Queues"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Populate(Writer, Consumer):
    def execute(self, contents, *args, **kwargs): self.write(contents, *args, **kwargs)
    def write(self, contents, *args, **kwargs): self.destination.put(contents, *args, **kwargs)


class Depopulate(Reader, Producer):
    def execute(self, *args, **kwargs): return self.read(*args, **kwargs)
    def read(self, *args, **kwargs): return self.source.get(*args, **kwargs)


class QueueMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.Variable = kwargs.get("variable", getattr(cls, "Variable", None))
        cls.Type = kwargs.get("type", getattr(cls, "Type", None))

    def __call__(cls, *args, capacity=None, contents=[], **kwargs):
        assert cls.Variable is not None
        assert cls.Type is not None
        assert isinstance(contents, list)
        assert (len(contents) <= capacity) if bool(capacity) else True
        parameters = dict(variable=cls.Variable)
        instance = cls.Type(maxsize=capacity if capacity is not None else 0)
        for content in contents:
            instance.put(content)
        wrapper = super(QueueMeta, cls).__call__(instance, *args, **parameters, **kwargs)
        return wrapper


class Queue(ABC, metaclass=QueueMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __bool__(self): return not self.empty
    def __len__(self): return self.size

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, instance, *args, variable, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__variable = variable
        self.__timeout = timeout
        self.__queue = instance

    @abstractmethod
    def get(self, *args, **kwargs): pass
    @abstractmethod
    def put(self, content, *args, **kwargs): pass

    @property
    def empty(self): return super().empty()
    @property
    def size(self): return super().qsize()

    @property
    def variable(self): return self.__variable
    @property
    def timeout(self): return self.__timeout
    @property
    def queue(self): return self.__queue


class StandardQueue(Queue):
    def get(self, *args, **kwargs):
        content = self.queue.get(timeout=self.timeout)
        self.queue.task_done()
        return content

    def put(self, content, *args, **kwargs):
        self.queue.put(content, timeout=self.timeout)


class PriorityQueue(Queue):
    def __init_subclass__(cls, *args, **kwargs):
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

    def get(self, *args, **kwargs):
        couple = self.queue.get(timeout=self.timeout)
        priority, content = couple
        self.queue.task_done()
        return content

    def put(self, content, *args, **kwargs):
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


class Stack(ABC):
    def __repr__(self): return f"{self.name}[{', '.join([variable for variable in self.queues.keys()])}]"
    def __getitem__(self, variable): return self.queues[variable]
    def __init__(self, *args, queues=[], **kwargs):
        assert isinstance(queues, list)
        assert all([isinstance(instance, Queue) for instance in Queue])
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__queues = {str(instance.variable): instance for instance in queues}

    @property
    def tables(self): return self.__tables
    @property
    def name(self): return self.__name


class Queues:
    FIFO = FIFOQueue
    LIFO = LIFOQueue
    HIPO = HIPOQueue
    LIPO = LIPOQueue



