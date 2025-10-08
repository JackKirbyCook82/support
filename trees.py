# -*- coding: utf-8 -*-
"""
Created on Mon Nov 4 2024
@name:   Tree Object
@author: Jack Kirby Cook

"""

from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.decorators import Dispatchers
from support.mixins import Mixin

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Node", "ParentalNode"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


Style = ntuple("Style", "branch terminate run blank")
class Styles:
    Double = Style("╠══", "╚══", "║  ", "   ")
    Single = Style("├──", "└──", "│  ", "   ")
    Curved = Style("├──", "╰──", "│  ", "   ")


def render(node, *args, style, layers=None, **kwargs):
    check = lambda indx, last: indx == last
    prefix = lambda indx, last: style.terminate if check(indx, last) else style.branch
    padding = lambda: "".join(style.blank if layer else style.run for layer in layers)
    create = lambda indx, last: padding() + prefix(indx, last)
    if layers is None: layers = []
    if not layers: yield "", node

    children = list(node.values())
    position = len(children) - 1
    for index, values in enumerate(children):
        sequence = values if isinstance(values, (list, tuple)) else [values]
        for value in sequence:
            yield create(index, position), value
            yield from render(value, *args, style=style, layers=[*layers, check(index, position)], **kwargs)


class Node(Mixin):
    def __init__(self, identity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__children = ODict()
        self.__identity = identity

    def __contains__(self, key): return bool(key in self.children.keys())
    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)
    def __reversed__(self): return reversed(self.items())
    def __iter__(self): return iter(self.items())
    def __str__(self): return self.render(style=Styles.Single)

    def keys(self): return self.children.keys()
    def values(self): return self.children.values()
    def items(self): return self.children.items()

    def transverse(self, *args, **kwargs):
        function = lambda value: [value] if isinstance(value, Node) else list(value)
        generator = (child for value in self.values() for child in function(value))
        for child in generator:
            assert isinstance(child, Node)
            yield child
            transverse = child.transverse(*args, **kwargs)
            yield from transverse

    def render(self, *args, style=Styles.Single, **kwargs):
        generator = self.renderer(*args, style=style, **kwargs)
        return "\n".join(list(generator))

    def renderer(self, *args, style=Styles.Single, **kwargs):
        generator = render(self, *args, style=style, **kwargs)
        for prefix, value in generator:
            label = str(value.identity) if isinstance(value, Node) else str(value)
            yield str(prefix) + str(label)

    def get(self, key): return self.children[key]
    def set(self, key, content): self.children[key] = content

    def extend(self, key, values):
        assert isinstance(values, list)
        assert all([isinstance(value, Node) for value in values])
        assert key in self.children.keys() and isinstance(self.children[key], list)
        self.children[key].extend(values)

    def append(self, key, value):
        assert isinstance(value, Node)
        assert key in self.children.keys() and isinstance(self.children[key], list)
        self.children[key].append(value)

    def expand(self, key):
        assert key in self.children.keys()
        assert isinstance(self.children[key], Node)
        child = self.children[key]
        self.children[key] = [child]

    def squeeze(self, key):
        assert key in self.children.keys()
        assert isinstance(self.children[key], list)
        assert len(self.children[key]) == 1
        child = self.children[key]
        self.children[key] = child

    @property
    def leafs(self): return [value for value in self.transverse() if not bool(value.children)]
    @property
    def branches(self): return [value for value in self.transverse() if bool(value.children)]
    @property
    def terminal(self): return not bool(self.children)
    @property
    def children(self): return self.__children
    @property
    def identity(self): return self.__identity


class ParentalNode(Node):
    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__parent = parent

    @Dispatchers.Type(locator=0)
    def assign(self, value):
        assert isinstance(value, Node)
        assert value.parent is None
        value.parent = self

    @assign.register(list)
    def assignments(self, values):
        for value in values: self.assign(value)

    def set(self, key, content):
        super().set(key, content)
        self.assign(content)

    def extend(self, key, values):
        super().extend(key, values)
        self.assign(values)

    def append(self, key, value):
        super().append(key, value)
        self.assign(value)

    @property
    def parent(self): return self.__parent
    @parent.setter
    def parent(self, parent): self.__parent = parent


