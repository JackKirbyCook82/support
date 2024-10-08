# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   Mixins Object
@author: Jack Kirby Cook

"""

import types
import inspect
import logging
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from functools import update_wrapper
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Node", "Fields", "Logging", "Emptying", "Sizing", "Pipelining", "Publisher", "Subscriber"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


Style = ntuple("Style", "branch terminate run blank")
aslist = lambda x: [x] if not isinstance(x, (list, tuple)) else list(x)
double = Style("╠══", "╚══", "║  ", "   ")
single = Style("├──", "└──", "│  ", "   ")
curved = Style("├──", "╰──", "│  ", "   ")


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


class Fields(object):
    def __init_subclass__(cls, *args, fields=[], **kwargs):
        assert all([attr not in fields for attr in ("fields", "keys", "values", "items")])
        existing = getattr(cls, "__fields__", [])
        update = [field for field in fields if field not in existing]
        cls.__fields__ = existing + update

    def __init__(self, *args, **kwargs):
        fields = list(self.__class__.__fields__)
        fields = ODict([(field, kwargs.get(field, None)) for field in fields])
        self.__fields = fields
        for field, content in fields.items():
            setattr(self, field, content)

    def tolist(self): return list(self.fields.items())
    def todict(self): return dict(self.fields)
    def items(self): return self.fields.items()
    def values(self): return self.fields.values()
    def keys(self): return self.fields.keys()

    @property
    def fields(self): return self.__fields


class Emptying(object):
    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(dict)
    def empty_mapping(self, mapping): return all([self.empty(value) for value in mapping.values()]) if bool(mapping) else True
    @empty.register(list)
    def empty_collection(self, collection): return all([self.empty(value) for value in collection]) if bool(collection) else True
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)


class Sizing(object):
    @typedispatcher
    def size(self, content): raise TypeError(type(content).__name__)
    @size.register(dict)
    def size_mapping(self, mapping): return sum([self.size(value) for value in mapping.values()])
    @size.register(list)
    def size_collection(self, collection): return sum([self.size(value) for value in collection])
    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.DataFrame)
    def size_dataframe(self, dataframe): return len(dataframe.dropna(how="all", inplace=False).index)
    @size.register(pd.Series)
    def size_series(self, series): return len(series.dropna(how="all", inplace=False).index)


class Logging(object):
    def __repr__(self): return str(self.name)
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.pop("name", self.__class__.__name__)
        self.__logger = __logger__

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


class Pipelining(ABC):
    def __new__(cls, *args, **kwargs):
        execute = cls.execute
        if not inspect.isgeneratorfunction(execute):
            def wrapper(self, *arguments, **parameters):
                assert isinstance(self, cls)
                results = execute(self, *arguments, **parameters)
                if results is not None: yield results
            update_wrapper(wrapper, execute)
            setattr(cls, "execute", wrapper)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        assert inspect.isgeneratorfunction(self.execute)
        assert inspect.isgeneratorfunction(self.generator)

    def __call__(self, *args, source=None, **kwargs):
        yield from self.wrapper(source, *args, **kwargs)

    @typedispatcher
    def wrapper(self, source, *args, **kwargs):
        yield from self.execute(source, *args, **kwargs)

    @wrapper.register(types.NoneType)
    def source(self, source, *args, **kwargs):
        assert isinstance(source, types.NoneType)
        yield from self.execute(*args, **kwargs)

    @wrapper.register(types.FunctionType)
    def function(self, function, *args, **kwargs):
        source = function(*args, **kwargs)
        yield from self.execute(source, *args, **kwargs)

    @wrapper.register(types.GeneratorType)
    def generator(self, generator, *args, **kwargs):
        for source in generator:
            yield from self.execute(source, *args, **kwargs)

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Node(object):
    def __init__(self, *args, **kwargs):
        self.__formatter = kwargs.get("formatter", lambda key, node: str(node.name))
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
    def leafs(self): return [value for value in self.transverse() if not bool(value.children)]
    @property
    def branches(self): return [value for value in self.transverse() if bool(value.children)]
    @property
    def children(self): return list(self.nodes.values())
    @property
    def size(self): return len(self.nodes)

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


class Publisher(object):
    def __init__(self, *args, **kwargs):
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


class Subscriber(ABC):
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__publishers = set()

    def observe(self, event, *args, publisher, **kwargs):
        assert isinstance(publisher, Publisher)
        self.reaction(event, *args, publisher=publisher, **kwargs)

    @abstractmethod
    def reaction(self, event, *args, publisher, **kwargs): pass
    @property
    def publishers(self): return self.__publishers
    @property
    def name(self): return self.__name









