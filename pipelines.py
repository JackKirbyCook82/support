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

from support.mixins import Locking
from support.dispatchers import kwargsdispatcher

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

    @kwargsdispatcher(key="file", func=lambda file: os.path.splitext(file))
    def read(self, *args, file, **kwargs):
        raise ValueError(file)

    @kwargsdispatcher(key="file", func=lambda file: os.path.splitext(file))
    def reader(self, *args, file, **kwargs):
        raise ValueError(file)

    @kwargsdispatcher(key="file", func=lambda file: os.path.splitext(file))
    def refer(self, *args, file, **kwargs):
        raise ValueError(file)

    @read.register("nc")
    def read_netcdf(self, *args, file, **kwargs):
        with self.locking(file):
            return xr.open_dataset(file, chunks=None)

    @refer.register("nc")
    def refer_netcdf(self, *args, file, partitions={}, **kwargs):
        assert isinstance(partitions, dict)
        with self.locking(file):
            return xr.open_dataset(file, chunks=partitions)

    @read.register("csv")
    def read_csv(self, *args, file, datatypes={}, datetypes=[], **kwargs):
        with self.locking(file):
            parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
            return pd.read_csv(file, iterator=False, **parms)

    @reader.register("csv")
    def reader_csv(self, *args, file, rows, datatypes={}, datetypes=[], **kwargs):
        with self.locking(file):
            parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
            return pd.read_csv(file, chucksize=rows, iterator=True, **parms)

    @refer.register("csv")
    def refer_csv(self, *args, file, size, datatypes={}, datetypes=[], **kwargs):
        with self.locking(file):
            parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
            return dk.read_csv(file, blocksize=size, **parms)

    @read.register("hdf")
    def read_hdf5(self, *args, file, group=None, **kwargs):
        with self.locking(file):
            return pd.read_hdf(file, key=group, iterator=False)

    @reader.register("hdf")
    def reader_hdf5(self, *args, file, group=None, rows, **kwargs):
        with self.locking(file):
            return pd.read_csv(file, key=group, chunksize=rows, iterator=True)


class Saver(Files, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, repository=repository, **kwargs)
        if not os.path.isdir(repository):
            os.mkdir(repository)

    @kwargsdispatcher(key="file", func=lambda file: os.path.splitext(file))
    def write(self, content, *args, file, **kwargs):
        raise ValueError(file)

    @write.register("nc")
    def write_netcdf(self, content, *args, file, mode, **kwargs):
        assert isinstance(content, xr.Dataset)
        with self.locking(file):
            xr.Dataset.to_netcdf(content, file, mode=mode, compute=True)

    @write.register("csv")
    def write_csv(self, content, *args, file, mode, **kwargs):
        assert isinstance(content, (pd.DataFrame, dk.DataFrame))
        with self.locking(file):
            parms = dict(index=False, header=True)
            if isinstance(content, dk.DataFrame):
                update = dict(compute=True, single_file=True, header_first_partition_only=True)
                parms.update(update)
            content.to_csv(file, mode=mode, **parms)

    @write.register("hdf")
    def write_hdf5(self, content, *args, file, group=None, mode, **kwargs):
        assert isinstance(content, (pd.DataFrame, dk.DataFrame))
        with self.locking(file):
            parms = dict(format="fixed", append=False)
            content.to_hdf(file, group, mode=mode, **parms)



