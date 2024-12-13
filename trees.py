# -*- coding: utf-8 -*-
"""
Created on Mon Nov 4 2024
@name:   Tree Object
@author: Jack Kirby Cook

"""

import logging
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SingleNode", "MultipleNode", "MixedNode"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


Style = ntuple("Style", "branch terminate run blank")
class Styles:
    Double = Style("╠══", "╚══", "║  ", "   ")
    Single = Style("├──", "└──", "│  ", "   ")
    Curved = Style("├──", "╰──", "│  ", "   ")


def render(node, *args, style, layers=[], **kwargs):
    last = lambda indx, length: indx == length
    prefix = lambda indx, length: style.terminate if last(indx, length) else style.blank
    padding = lambda: "".join([style.blank if layer else style.run for layer in layers])
    function = lambda indx, length: "".join([padding(), prefix(indx, length)])
    if not layers: yield "", node
    children = iter(node.items())
    size = len(list(children))
    for index, (key, values) in enumerate(children):
        for value in [values] if not isinstance(values, (list, tuple)) else list(values):
            yield function(index, size - 1), value
            yield from render(value, *args, layers=[*layers, last(index, size - 1)], style=style, **kwargs)


class Node(ABC):
    def __init__(self, *args, **kwargs):
        self.__nodes = ODict()

    def __contains__(self, key): return bool(key in self.nodes.keys())
    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)

    def __reversed__(self): return reversed(self.items())
    def __iter__(self): return iter(self.items())

    def keys(self): return self.nodes.keys()
    def values(self): return self.nodes.values()
    def items(self): return self.nodes.items()

    def render(self, *args, style=Styles.Single, **kwargs):
        generator = render(self, style=style)
        rows = [prefix + str() for prefix, value in iter(generator)]
        return "\n".format(rows)

    @property
    def leafs(self): return [value for value in self.transverse() if not bool(value.children)]
    @property
    def branches(self): return [value for value in self.transverse() if bool(value.children)]
    @property
    def children(self): return list(self.nodes.values())
    @property
    def size(self): return len(self.nodes)

    @abstractmethod
    def transverse(self, *args, **kwargs): pass
    @abstractmethod
    def set(self, *args, **kwargs): pass
    @abstractmethod
    def get(self, *args, **kwargs): pass

    @property
    def nodes(self): return self.__nodes


class SingleNode(Node):
    def get(self, key): return self.nodes[key]
    def set(self, key, value):
        assert isinstance(value, Node)
        self.nodes[key] = value

    def transverse(self, *args, **kwargs):
        for value in self.values():
            assert isinstance(value, Node)
            yield value
            transverse = value.transverse(*args, **kwargs)
            yield from transverse


class MultipleNode(Node):
    def get(self, key): return self.nodes[key]
    def set(self, key, value):
        assert isinstance(value, (list, Node))
        if key not in self.nodes: self.nodes[key] = []
        if isinstance(value, list): self.nodes[key].extend(value)
        else: self.nodes[key].append(value)

    def transverse(self, *args, **kwargs):
        for values in self.values():
            assert isinstance(values, list)
            for value in values:
                yield value
                transverse = value.transverse(*args, **kwargs)
                yield from transverse


class MixedNode(Node):
    def get(self, key): return self.nodes[key]
    def set(self, key, content):
        assert isinstance(content, (Node, list))
        self.nodes[key] = content

    def append(self, key, value):
        assert isinstance(value, Node)
        if key not in self.nodes.keys: self.nodes[key] = []
        self.nodes[key].append(value)

    def extend(self, key, values):
        assert isinstance(values, list) and all([isinstance(value, Node) for value in values])
        if key not in self.nodes.keys(): self.nodes[key] = []
        self.nodes[key].extend(values)

    def transverse(self, *args, **kwargs):
        for content in self.values():
            assert isinstance(content, (list, Node))
            values = [content] if isinstance(content, Node) else values
            for value in content:
                yield value
                transverse = value.transverse(*args, **kwargs)
                yield from transverse


