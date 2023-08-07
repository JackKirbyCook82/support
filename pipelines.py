# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Pipeline Objects
@author: Jack Kirby Cook

"""

import os
import time
import types
import inspect
import logging
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from functools import reduce
from abc import ABC, abstractmethod

from utilities.mixins import Locking
from utilities.dispatchers import typedispatcher, valuedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Processor", "Calculator", "Downloader", "Uploader", "Saver", "Loader"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class Pipeline(list):
    def __repr__(self): return "|".join(list(map(repr, self)))
    def __init__(self, processors):
        assert isinstance(processors, list)
        assert all([isinstance(processor, Processor) for processor in processors])
        super().__init__(processors)

    def __add__(self, other):
        assert isinstance(other, Processor)
        return Pipeline([*self, other])

    def __call__(self, *args, **kwargs):
        source, segments = self[0](*args, **kwargs), self[1:]
        assert isinstance(source, types.GeneratorType)
        assert all([inspect.isgeneratorfunction(segment.__call__) for segment in segments])
        generator = reduce(lambda inner, outer: outer(inner, *args, **kwargs), segments, source)
        yield from iter(generator)


class Processor(ABC):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __add__(self, other):
        assert isinstance(other, Processor)
        return Pipeline([self, other])

    def __call__(self, *args, **kwargs):
        source = args[0] if bool(args) and isinstance(args[0], types.GeneratorType) else None
        process = self.process(source, *args, **kwargs) if source is not None else self.generator(*args, **kwargs)
        yield from iter(process)

    def process(self, source, *args, **kwargs):
        for content in iter(source):
            generator = self.generator(content, *args, **kwargs)
            yield from iter(generator)

    def generator(self, *args, **kwargs):
        start = time.time()
        if not inspect.isgeneratorfunction(self.execute):
            self.execute(*args, **kwargs)
            LOGGER.info("Processed: {}|{:.2f}s".format(repr(self), time.time() - start))
            return
        generator = self.execute(*args, **kwargs)
        assert isinstance(generator, types.GeneratorType)
        for content in iter(generator):
            LOGGER.info("Processed: {}|{:.2f}s".format(repr(self), time.time() - start))
            yield content
            start = time.time()

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def name(self): return self.__name


class Calculator(Processor, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        calculations = [calculation for calculation in getattr(cls, "__calculations__", [])]
        calculations.extend(kwargs.get("calculations", []))
        cls.__calculations__ = calculations

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        calculations = list(self.__class__.__calculations__)
        calculations = [calculation(*args, **kwargs) for calculation in iter(calculations)]
        self.__calculations = calculations

    @property
    def calculations(self): return self.__calculations


class Websites(Processor, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        pages = {key: value for key, value in getattr(cls, "__pages__", {}).items()}
        pages.update(kwargs.get("pages", {}))
        cls.__pages__ = pages

    def __getitem__(self, key): return self.pages[key]
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        pages = list(self.__class__.__pages__.items())
        pages = {key: page for key, page in iter(pages)}
        pages = {key: page(source) for key, page in pages.items()}
        self.__pages = pages

    @property
    def pages(self): return self.__pages


class Downloader(Websites, ABC): pass
class Uploader(Websites, ABC): pass


class Files(Processor, Locking, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, **kwargs)
        self.__repository = repository

    @property
    def repository(self): return self.__repository


class Loader(Files, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, repository=repository, **kwargs)
        if not os.path.isdir(repository):
            raise FileNotFoundError(repository)

    @valuedispatcher
    def read(self, datatype, *args, **kwargs): raise ValueError(datatype.__name__)
    @valuedispatcher
    def refer(self, datatype, *args, **kwargs): raise ValueError(datatype.__name__)

    @read.register(xr.Dataset)
    def read_dataset(self, *args, file, **kwargs):
        with self.locking(file):
            return xr.open_dataset(file, chunks=None)

    @read.register(pd.DataFrame)
    def read_dataframe(self, *args, file, rows=None, datatypes={}, datetypes=[], **kwargs):
        with self.locking(file):
            iterator = bool(rows is not None)
            parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
            return pd.read_csv(file, chucksize=rows, iterator=iterator, **parms)

    @refer.register(xr.Dataset)
    def refer_dataset(self, *args, file, partitions={}, **kwargs):
        with self.locking(file):
            return xr.open_dataset(file, chunks=partitions)

    @refer.register(pd.DataFrame)
    def refer_dataframe(self, *args, file, size=None, datatypes={}, datetypes=[], **kwargs):
        with self.locking(file):
            parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
            return dk.read_csv(file, blocksize=size, **parms)


class Saver(Files, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, repository=repository, **kwargs)
        if not os.path.isdir(repository):
            os.mkdir(repository)

    @typedispatcher
    def write(self, content, *args, **kwargs): raise TypeError(type(content).__name__)

    @write.register(xr.Dataset)
    def write_dataset(self, content, *args, file, mode, **kwargs):
        with self.locking(file):
            xr.Dataset.to_netcdf(content, file, mode=mode, compute=True)

    @write.register(pd.DataFrame)
    def write_dataframe(self, content, *args, file, mode, rows=None, **kwargs):
        with self.locking(file):
            parms = dict(index=False, header=True)
            pd.DataFrame.to_csv(content, file, mode=mode, chunksize=rows, **parms)

    @write.register(dk.DataFrame)
    def write_daskframe(self, content, *args, file, mode, rows=None, **kwargs):
        with self.locking(file):
            parms = dict(index=False, header=True, single_file=True, header_first_partition_only=True)
            dk.DataFrame.to_csv(content, file, mode=mode, chunksize=rows, compute=True, **parms)



