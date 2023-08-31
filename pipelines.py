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
from support.dispatchers import typedispatcher, valuedispatcher

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

    def read(self, *args, file, **kwargs):
        extension = os.path.splitext(file)
        with self.locking(file):
            pass

    def refer(self, *args, file, **kwargs):
        extension = os.path.splitext(file)
        with self.locking(file):
            pass

    @valuedispatcher
    def reader(self, extension, *args, file, **kwargs): raise ValueError(extension)
    @valuedispatcher
    def referer(self, extension, *args, file, **kwargs): raise ValueError(extension)

    @reader.register("csv")
    def csv_reader(self, *args, file, **kwargs):
        pass

    @reader.register("hdf5")
    def hdf5_reader(self, *args, file, **kwargs):
        pass

    @reader.register("netcdf")
    def netcdf_reader(self, *args, file, **kwargs):
        pass

    @reader.register("csv")
    def csv_referer(self, *args, file, **kwargs):
        pass

    @reader.register("hdf5")
    def hdf5_referer(self, *args, file, **kwargs):
        pass

    @reader.register("netcdf")
    def netcdf_referer(self, *args, file, **kwargs):
        pass


class Saver(Files, ABC):
    def __init__(self, *args, repository, **kwargs):
        super().__init__(*args, repository=repository, **kwargs)
        if not os.path.isdir(repository):
            os.mkdir(repository)

    def write(self, content, *args, file, mode, **kwargs):
        with self.locking(file):
            pass

    @typedispatcher
    def csv(self, content, *args, file, mode, **kwargs): raise TypeError(type(content).__name__)
    @typedispatcher
    def hdf5(self, content, *args, file, mode, **kwargs): raise TypeError(type(content).__name__)
    @typedispatcher
    def netcdf(self, content, *args, file, mode, **kwargs): raise TypeError(type(content).__name__)

    @csv.register(pd.DataFrame)
    def csv_pd(self, content, *args, file, mode, **kwargs):
        pass

    @csv.register(dk.DataFrame)
    def csv_dk(self, content, *args, file, mode, **kwargs):
        pass

    @hdf5.register(pd.DataFrame)
    def hdf5_pd(self, content, *args, file, mode, **kwargs):
        pass

    @hdf5.register(dk.DataFrame)
    def hdf5_dk(self, content, *args, file, mode, **kwargs):
        pass

    @netcdf.register(xr.Dataset)
    def netcdf_xr(self, content, *args, file, mode, **kwargs):
        pass



