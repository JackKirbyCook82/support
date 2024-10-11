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
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict
from functools import update_wrapper, total_ordering, reduce

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Node", "Query", "Field", "Logging", "Emptying", "Sizing", "Sourcing", "Pipelining", "Publisher", "Subscriber"]
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


@total_ordering
class Field(ABC):
    pass


@total_ordering
class Query(ABC):
    def __init_subclass__(cls, *args, fields=[], **kwargs):
        assert all([attr not in fields for attr in ("fields", "contents")])
        existing = getattr(cls, "fields", [])
        update = [field for field in fields if field not in existing]
        cls.fields = tuple(existing + update)

    def __new__(cls, contents, *args, **kwargs):
        assert isinstance(contents, (tuple, list))
        assert len(cls.fields) == len(contents)
        instance = super().__new__(cls)
        setattr(instance, "fields", cls.fields)
        for field, content in zip(cls.fields, contents):
            setattr(instance, field, content)
        return instance

    def __iter__(self): return ((field, content) for field, content in zip(self.fields, self.contents))
    def __init__(self, contents, *args, delimiter="|", **kwargs):
        self.contents = tuple(contents)
        self.delimiter = str(delimiter)

    def __hash__(self): return hash(tuple([hash(content) for content in self.contents]))
    def __str__(self): return str(self.delimiter).join([str(content) for content in self.contents])

    def __eq__(self, other):
        assert type(other) is type(self) and self.fields == other.fields
        return all([primary == secondary for primary, secondary in zip(self.contents, other.contents)])

    def __lt__(self, other):
        assert type(other) is type(self) and self.fields == other.fields
        return all([primary < secondary for primary, secondary in zip(self.contents, other.contents)])

    @classmethod
    def fromstring(cls, string, delimiter): return cls([field.fromstring(content) for field, content in zip(cls.fields, str(string).split(delimiter))])
    def tostring(self, delimiter): return str(delimiter).join([str(content) for content in self.contents])


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
    @empty.register(types.NoneType)
    def empty_null(self, *args, **kwargs): return False


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
    @size.register(types.NoneType)
    def size_nothing(self, *args, **kwargs): return 0


class Sourcing(object):
    @typedispatcher
    def source(self, source, *args, query, **kwargs):
        assert issubclass(query, Query)
        yield query(source)

    @source.register(list)
    def source_collection(self, collection, *args, query, **kwargs):
        assert issubclass(query, Query)
        for content in collection:
            yield query(content)

    @source.register(pd.DataFrame)
    def source_dataframe(self, dataframe, *args, query, **kwargs):
        assert issubclass(query, Query)
        generator = dataframe.groupby(query.fields)
        for content, dataframe in iter(generator):
            yield query(content), dataframe

    @source.register(xr.Dataset)
    def source_dataset(self, dataset, *args, query, **kwargs):
        assert issubclass(query, Query)
        for field in iter(query.fields):
            dataset = dataset.expand_dims(field)
        dataset = dataset.stack(stack=query.fields)
        generator = dataset.groupby("stack")
        for content, dataset in iter(generator):
            dataset = dataset.unstack().drop_vars("stack")
            yield query(content), dataset

    @staticmethod
    def align(source, query):
        assert isinstance(source, (pd.DataFrame, xr.Dataset)) and isinstance(query, Query)
        mask = [source[field] == content for field, content in iter(query)]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        return source.where(mask).dropna(how="all", inplace=False)


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
        generator = self.generator(source, *args, **kwargs)
        yield from generator

    @typedispatcher
    def generator(self, source, *args, **kwargs):
        generator = self.execute(source, *args, **kwargs)
        yield from generator

    @generator.register(types.NoneType)
    def generator_empty(self, source, *args, **kwargs):
        assert isinstance(source, types.NoneType)
        generator = self.execute(*args, **kwargs)
        yield from generator

    @generator.register(types.FunctionType)
    def generator_function(self, source, *args, **kwargs):
        content = source(*args, **kwargs)
        generator = self.execute(content, *args, **kwargs)
        yield from generator

    @generator.register(types.GeneratorType)
    def generator_generator(self, source, *args, **kwargs):
        for content in iter(source):
            generator = self.execute(content, *args, **kwargs)
            yield from generator

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Logging(object):
    def __repr__(self): return str(self.name)
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.pop("name", self.__class__.__name__)
        self.__logger = __logger__

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


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









