# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import logging
import multiprocessing
import numpy as np
import pandas as pd
import dask.dataframe as dk
from enum import IntEnum
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple

from support.pipelines import Producer, Consumer
from support.dispatchers import kwargsdispatcher
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "File", "FileTypes", "FileTimings"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


FileTypes = IntEnum("Typing", ["NC", "HDF", "CSV"], start=1)
FileTimings = IntEnum("Timing", ["EAGER", "LAZY"], start=1)
FileMethod = ntuple("FileMethod", "filetype filetiming")

csv_eager = FileMethod(FileTypes.CSV, FileTimings.EAGER)
csv_lazy = FileMethod(FileTypes.CSV, FileTimings.LAZY)
hdf_eager = FileMethod(FileTypes.HDF, FileTimings.EAGER)
hdf_lazy = FileMethod(FileTypes.HDF, FileTimings.LAZY)
nc_eager = FileMethod(FileTypes.NC, FileTimings.EAGER)
nc_lazy = FileMethod(FileTypes.NC, FileTimings.LAZY)


class Loader(Producer):
    def execute(self, *args, **kwargs):
        pass


class Saver(Consumer):
    def execute(self, contents, *args, **kwargs):
        pass


class Lock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class DataStream(ABC):
    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass
    @staticmethod
    @abstractmethod
    def empty(content): pass


class DataframeStream(DataStream, datatype=pd.DataFrame):
    def __init__(self, *args, header={}, **kwargs):
        assert isinstance(header, dict)
        header = [(key, value) for key, value in header.items()]
        self.__types = {key: value for (key, value) in iter(header) if not any([value is str, value is np.datetime64])}
        self.__dates = [key for (key, value) in iter(header) if value is np.datetime64]

    @kwargsdispatcher("method")
    def load(self, *args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())
    @kwargsdispatcher("method")
    def save(self, dataframe, args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())

    @load.register.value(csv_eager)
    def load_eager_csv(self, *args, file, **kwargs):
        return pd.read_csv(file, iterator=False, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @load.register.value(csv_lazy)
    def load_lazy_csv(self, *args, file, size, **kwargs):
        return dk.read_csv(file, blocksize=size, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @save.register.value(csv_eager)
    def save_eager_csv(self, dataframe, *args, file, mode, **kwargs):
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @save.register.value(csv_lazy)
    def save_lazy_csv(self, dataframe, *args, file, mode, **kwargs):
        parameters = dict(compute=True, single_file=True, header_first_partition_only=True)
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w", **parameters)

    @staticmethod
    def empty(dataframe): return bool(dataframe.empty)
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates


class FileMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is FileMeta for base in cls.__bases__]):
            return
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__header__ = kwargs.get("header", getattr(cls, "__header__", None))
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__variable__ is not None
        assert cls.__datatype__ is not None
        assert cls.__header__ is not None
        assert cls.__query__ is not None
        instance = super(FileMeta, cls).__call__(*args, mutex=Lock(), **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, repository, mutex, filetype, filetiming, **kwargs):
        if not os.path.exists(repository):
            os.mkdir(repository)
        query, querytype = self.__class__.__query__
        self.__variable = self.__class__.__variable__
        self.__datatype = self.__class__.__datatype__
        self.__header = self.__class__.__header__
        self.__repository = repository
        self.__filetiming = filetiming
        self.__filetype = filetype
        self.__querytype = querytype
        self.__query = query
        self.__mutex = mutex

    @property
    def repository(self): return self.__repository
    @property
    def variable(self): return self.__variable
    @property
    def querytype(self): return self.__querytype
    @property
    def datatype(self): return self.__datatype
    @property
    def filetype(self): return self.__filetype
    @property
    def header(self): return self.__header
    @property
    def timing(self): return self.__timing
    @property
    def query(self): return self.__query
    @property
    def mutex(self): return self.__mutex



