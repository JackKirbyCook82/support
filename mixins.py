# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   Mixins
@author: Jack Kirby Cook

"""

import multiprocessing
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Locking", "Node", "Publisher", "Subscriber"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = ""


Style = ntuple("Style", "branch terminate run blank")
aslist = lambda x: [x] if not isinstance(x, (list, tuple)) else list(x)
double = Style("╠══", "╚══", "║  ", "   ")
single = Style("├──", "└──", "│  ", "   ")
curved = Style("├──", "╰──", "│  ", "   ")


class Mixin(object):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()


class Locking(Mixin):
    locks = {}

    @classmethod
    def locking(cls, key):
        if key not in cls.locks.keys():
            cls.locks[key] = multiprocessing.Lock()
        return cls.locks[key]

    @classmethod
    def lock(cls, key):
        if key not in cls.locks.keys():
            cls.locks[key] = multiprocessing.Lock()
        cls.locks[key].acquire()

    @classmethod
    def unlock(cls, key):
        if key not in cls.locks.keys():
            return
        cls.locks[key].release()


def renderer(node, layers=[], style=single):
    assert hasattr(node, "children") and hasattr(node, "size")
    last = lambda i, x: i == x
    func = lambda i, x: "".join([pads(), pre(i, x)])
    pre = lambda i, x: style.terminate if last(i, x) else style.blank
    pads = lambda: "".join([style.blank if layer else style.run for layer in layers])
    if not layers:
        yield "", None, node
    for index, (key, values) in enumerate(iter(node.children)):
        for value in aslist(values):
            yield func(index, node.size - 1), key, value
            yield from renderer(value, layers=[*layers, last(index, node.size - 1)], style=style)


class Node(Mixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__formatter = kwargs.get("formatter", lambda key, value: str(value.name))
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__style = kwargs.get("style", single)
        self.__nodes = ODict()

    def set(self, key, value): self.nodes[key] = value
    def get(self, key): return self.nodes[key]

    def append(self, key, value):
        assert isinstance(value, Node)
        self.nodes[key] = aslist(self.nodes.get(key, []))
        self.nodes[key].append(value)

    def extend(self, key, value):
        assert isinstance(value, list)
        self.nodes[key] = aslist(self.nodes.get(key, []))
        self.nodes[key].extend(value)

    def keys(self): return self.nodes.keys()
    def values(self): return self.nodes.values()
    def items(self): return self.nodes.items()

    def transverse(self):
        for value in self.values():
            yield value
            yield from value.transverse()

    @property
    def size(self): return len(self.nodes)
    @property
    def children(self): return list(self.nodes.values())
    @property
    def sources(self): return [value for value in self.transverse() if not value.children]

    @property
    def tree(self):
        generator = renderer(self, style=self.style)
        rows = [pre + self.formatter(key, value) for pre, key, value in generator]
        return "\n".format(rows)

    @property
    def formatter(self): return self.__formatter
    @property
    def style(self): return self.__style
    @property
    def nodes(self): return self.__nodes
    @property
    def name(self): return self.__name


class Publisher(Mixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__subscribers = set()

    def register(self, *subscribers):
        assert all([isinstance(subscriber, Subscriber) for subscriber in subscribers])
        for subscriber in subscribers:
            self.subscribers.add(subscriber)
            subscriber.publishers.add(self)

    def unregister(self, *subscribers):
        assert all([isinstance(subscriber, Subscriber) for subscriber in subscribers])
        for subscriber in subscribers:
            self.subscribers.discard(subscriber)
            subscriber.publishers.discard(self)

    def publish(self, event, *args, **kwargs):
        for subscriber in self.subscribers:
            subscriber.observe(event, *args, publisher=self, **kwargs)

    @property
    def subscribers(self): return self.__subscribers
    @property
    def name(self): return self.__name


class Subscriber(Mixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__publisher = set()

    def observe(self, event, *args, publisher, **kwargs):
        assert isinstance(publisher, Publisher)
        self.reaction(event, *args, publisher=publisher, **kwargs)

    @abstractmethod
    def reaction(self, event, *args, publisher, **kwargs): pass
    @property
    def publishers(self): return self.__publishers
    @property
    def name(self): return self.__name






