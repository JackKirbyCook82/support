# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Mixins Object
@author: Jack Kirby Cook

"""

import time
import types
import logging
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from collections import OrderedDict as ODict

from support.decorators import Dispatchers

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Mixin", "Naming", "Logging", "Emptying", "Memory", "Sizing", "Partition", "Publisher", "Subscriber", "Delayer"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Mixin(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()

    def __new__(cls, *args, **kwargs):
        try: return super().__new__(cls, *args, **kwargs)
        except TypeError: return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()


class Naming(Mixin):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        fields = getattr(cls, "fields", []) + kwargs.get("fields", [])
        assert "fields" not in fields
        cls.fields = fields

    def __iter__(self): return iter(self.fields.items())
    def __getitem__(self, field): return self.fields[field]
    def __getattr__(self, field):
        if field in self.fields.keys(): return self.fields[field]
        raise AttributeError(field)

    def __new__(cls, *args, **kwargs):
        fields = {field: kwargs.get(field, None) for field in cls.fields}
        instance = super().__new__(cls)
        for key, value in fields.items(): setattr(instance, key, value)
        setattr(instance, "fields", fields)
        return instance

    def items(self): return [(field, getattr(self, field)) for field in self.fields]
    def values(self): return [getattr(self, field) for field in self.fields]
    def keys(self): return list(self.fields)


class Logging(Mixin):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__title__ = kwargs.get("title", getattr(cls, "__title__", None))

    def __repr__(self): return str(self.name)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__title = kwargs.get("title", self.__class__.__title__)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__logger = __logger__

    def console(self, *strings, **parameters):
        title = parameters.get("title", self.title)
        string = "|".join(list(strings))
        if not bool(string): string = f"{str(title)}[{repr(self)}]"
        else: string = f"{str(title)}[{repr(self)}]:  {str(string)}"
        self.logger.info(string)

    @property
    def logger(self): return self.__logger
    @property
    def title(self): return self.__title
    @property
    def name(self): return self.__name


class Keys(Mixin):
    @Dispatchers.Type(locator=0)
    def keys(self, contents, *args, **kwargs): raise TypeError(type(contents))

    @keys.register(list)
    def __collection(self, collection, *args, **kwargs):
        for content in iter(collection):
            yield from self.keys(content, *args, **kwargs)

    @keys.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, by, **kwargs):
        for group in dataframe.groupby(list(by)).groups.keys():
            if not isinstance(group, tuple): group = [group]
            if callable(by): group = by(list(group))
            yield group

    @keys.register(xr.Dataset)
    def __dataset(self, dataset, *args, by, **kwargs):
        dataset = dataset.stack(stack=list(by))
        for group in dataset.groupby("stack").groups.keys():
            if not isinstance(group, tuple): group = [group]
            if callable(by): group = by(list(group))
            yield group


class Values(Mixin):
    @Dispatchers.Type(locator=0)
    def values(self, contents, *args, **kwargs): raise TypeError(type(contents))

    @values.register(list)
    def __collection(self, collection, *args, **kwargs):
        for content in iter(collection):
            yield from self.values(content, *args, **kwargs)

    @values.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, by, **kwargs):
        generator = dataframe.groupby(list(by))
        for values, dataframe in iter(generator):
            yield dataframe

    @values.register(xr.Dataset)
    def __dataset(self, dataset, *args, by, **kwargs):
        dataset = dataset.stack(stack=list(by))
        generator = dataset.groupby("stack")
        for values, dataset in iter(generator):
            dataset = dataset.unstack().drop_vars("stack")
            yield dataset


class Partition(Keys, Values):
    @Dispatchers.Type(locator=0)
    def partition(self, contents, *args, **kwargs): raise TypeError(type(contents))

    @partition.register(list)
    def __collection(self, collection, *args, **kwargs):
        for content in iter(collection):
            yield from self.partition(content, *args, **kwargs)

    @partition.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, by, **kwargs):
        generator = dataframe.groupby(list(by))
        for values, dataframe in iter(generator):
            if not isinstance(values, tuple): values = [values]
            partition = ODict(zip(list(by), values))
            if callable(by): partition = by(partition)
            yield partition, dataframe

    @partition.register(xr.Dataset)
    def __dataset(self, dataset, *args, by, **kwargs):
        dataset = dataset.stack(stack=list(by))
        generator = dataset.groupby("stack")
        for values, dataset in iter(generator):
            if not isinstance(values, tuple): values = [values]
            dataset = dataset.unstack().drop_vars("stack")
            partition = ODict(zip(list(by), values))
            if callable(by): partition = by(partition)
            yield partition, dataset


class Publisher(Mixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class Subscriber(Mixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__publishers = set()

    def observe(self, event, *args, publisher, **kwargs):
        assert isinstance(publisher, Publisher)
        self.reaction(event, *args, publisher=publisher, **kwargs)

    @abstractmethod
    def reaction(self, event, *args, publisher, **kwargs): pass
    @property
    def publishers(self): return self.__publishers


class Emptying(Mixin):
    @Dispatchers.Type(locator=0)
    def empty(self, content, *args, **kwargs): raise TypeError(type(content))
    @empty.register(dict)
    def __mapping(self, mapping, *args, **kwargs): return all([self.empty(value, *args, **kwargs) for value in mapping.values()]) if bool(mapping) else True
    @empty.register(list)
    def __collection(self, collection, *args, **kwargs): return all([self.empty(value, *args, **kwargs) for value in collection]) if bool(collection) else True
    @empty.register(xr.Dataset)
    def __dataset(self, dataset, key, *args, **kwargs): return self.empty([dataset[key]], *args, **kwargs)
    @empty.register(xr.DataArray)
    def __dataarray(self, dataarray, *args, **kwargs): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, **kwargs): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def __series(self, series, *args, **kwargs): return bool(series.empty)
    @empty.register(types.NoneType)
    def __nothing(self, *args, **kwargs): return True


class Sizing(Mixin):
    @Dispatchers.Type(locator=0)
    def size(self, content, *args, **kwargs): raise TypeError(type(content))
    @size.register(dict)
    def __mapping(self, mapping, *args, **kwargs): return sum([self.size(value, *args, **kwargs) for value in mapping.values()])
    @size.register(list)
    def __collection(self, collection, *args, **kwargs): return sum([self.size(value, *args, **kwargs) for value in collection])
    @size.register(xr.Dataset)
    def __dataset(self, dataset, key, *args, **kwargs): return self.size([dataset[key]], *args, **kwargs)
    @size.register(xr.DataArray)
    def __dataarray(self, dataarray, *args, **kwargs): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.DataFrame)
    def __dataframe(self, dataframe, *args, **kwargs): return len(dataframe.dropna(how="all", inplace=False).index)
    @size.register(pd.Series)
    def __series(self, series, *args, **kwargs): return len(series.dropna(how="all", inplace=False).index)
    @size.register(types.NoneType)
    def __nothing(self, *args, **kwargs): return 0


class Memory(Mixin):
    @Dispatchers.Type(locator=0)
    def memory(self, content, *args, **kwargs): raise TypeError(type(content))
    @memory.register(dict)
    def __mapping(self, mapping, *args, **kwargsg): return sum([self.memory(value) for value in mapping.values()])
    @memory.register(list)
    def __collection(self, collection, *args, **kwargs): return sum([self.memory(value) for value in collection])
    @memory.register(pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    def __content(self, content, *args, **kwargs): return content.nbytes
    @memory.register(types.NoneType)
    def __nothing(self, *args, **kwargs): return 0


class Delayer(Logging, title="Waiting"):
    def __init__(self, delay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(delay, int)
        self.__timer = time.monotonic()
        self.__delay = int(delay)

    def __enter__(self):
        elapsed = time.monotonic() - self.timer
        delay = max(self.delay - elapsed, 0)
        self.console(f"{delay:.2f} seconds")
        time.sleep(delay)
        return self

    def __exit__(self, error_type, error_value, error_traceback):
        self.timer = time.monotonic()

    @property
    def delay(self): return self.__delay
    @property
    def timer(self): return self.__timer
    @timer.setter
    def timer(self, timer): self.__timer = timer







