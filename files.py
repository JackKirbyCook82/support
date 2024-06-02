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
from support.meta import SingletonMeta, AttributeMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "FileDirectory", "FileQuery", "FileData", "FileTyping", "FileTiming"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


FileTyping = IntEnum("Typing", ["NC", "HDF", "CSV"], start=1)
FileTiming = IntEnum("Timing", ["EAGER", "LAZY"], start=1)
FileMethod = ntuple("Method", "typing timing")

csv_eager = FileMethod(FileTyping.CSV, FileTiming.EAGER)
csv_lazy = FileMethod(FileTyping.CSV, FileTiming.LAZY)
hdf_eager = FileMethod(FileTyping.HDF, FileTiming.EAGER)
hdf_lazy = FileMethod(FileTyping.HDF, FileTiming.LAZY)
nc_eager = FileMethod(FileTyping.NC, FileTiming.EAGER)
nc_lazy = FileMethod(FileTyping.NC, FileTiming.LAZY)


class Loader(Producer):
    def __init__(self, *args, source={}, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        assert isinstance(source, dict) and all([isinstance(file, FileDirectory) for file in source.keys()])
        self.__source = source

    def execute(self, *args, **kwargs):
        file = list(self.source.keys())[0]
        for variable, query, extension in iter(file):
            contents = self.read(*args, query=query, **kwargs)
            contents = {variable: content for variable, content in contents.items() if content is not None}
            yield {str(file.query): str(query)} | contents

    def read(self, *args, query, **kwargs):
        contents = {str(file.variable): file.read(*args, query=query, mode=mode, **kwargs) for file, mode in self.source.items()}
        return contents

    @property
    def source(self): return self.__source


class Saver(Consumer):
    def __init__(self, *args, destination, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        assert isinstance(destination, dict) and all([isinstance(file, FileDirectory) for file in destination.keys()])
        self.__destination = destination

    def execute(self, contents, *args, **kwargs):
        contents = {variable: content for variable, content in contents.items() if content is not None}
        self.write(contents, *args, **kwargs)

    def write(self, contents, *args, **kwargs):
        for file, mode in self.destination.items():
            query = contents[str(file.query)]
            content = contents[str(file.variable)]
            file.write(content, *args, query=query, mode=mode, **kwargs)

    @property
    def destination(self): return self.__destination


class FileQuery(ntuple("Query", "name tofile fromfile")):
    def __str__(self): return str(self.name)


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileDirectoryMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))
        cls.__data__ = kwargs.get("data", getattr(cls, "__data__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__variable__ is not None
        assert cls.__query__ is not None
        assert cls.__data__ is not None
        mutex = FileLock()
        instance = super(FileDirectoryMeta, cls).__call__(*args, mutex=mutex, **kwargs)
        return instance


class FileDirectory(ABC, metaclass=FileDirectoryMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __hash__(self): return hash(str(self.variable))
    def __init__(self, *args, repository, mutex, typing, timing, **kwargs):
        if not os.path.exists(repository):
            os.mkdir(repository)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__variable = self.__class__.__variable__
        self.__query = self.__class__.__query__
        self.__data = self.__class__.__data__
        self.__repository = repository
        self.__timing = timing
        self.__typing = typing
        self.__mutex = mutex

    def __iter__(self):
        directory = os.path.join(self.repository, self.variable)
        variable = str(self.variable)
        for file in list(os.listdir(directory)):
            filename, extension = str(file).split(".")
            query = self.query.fromfile(filename)
            yield variable, query, extension

    def read(self, *args, query, mode, **kwargs):
        method = FileMethod(self.typing, self.timing)
        directory = os.path.join(self.repository, self.variable)
        filename = self.query.tofile(query)
        file = ".".join([filename, str(self.typing.name).lower()])
        filepath = os.path.join(directory, file)
        if not os.path.exists(filepath):
            return
        with self.mutex[filepath]:
            parameters = dict(file=filepath, mode=mode, method=method)
            content = self.data.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, query, mode, **kwargs):
        assert content is not None
        method = FileMethod(self.typing, self.timing)
        directory = os.path.join(self.repository, self.variable)
        filename = self.query.tofile(query)
        file = ".".join([filename, str(self.typing.name).lower()])
        filepath = os.path.join(directory, file)
        with self.mutex[filepath]:
            parameters = dict(file=filepath, mode=mode, method=method)
            self.data.save(content, *args, **parameters, **kwargs)
        __logger__.info("Saved: {}".format(str(filepath)))

    @property
    def repository(self): return self.__repository
    @property
    def variable(self): return self.__variable
    @property
    def timing(self): return self.__timing
    @property
    def typing(self): return self.__typing
    @property
    def query(self): return self.__query
    @property
    def mutex(self): return self.__mutex
    @property
    def data(self): return self.__data
    @property
    def name(self): return self.__name


class FileDataMeta(AttributeMeta): pass
class FileData(ABC, metaclass=FileDataMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass
    @abstractmethod
    def empty(self, content): pass


class FileDataFrame(FileData, attribute="Dataframe"):
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
    def load_eager_csv(self, *args, file, mode, **kwargs):
        assert mode == "r"
        return pd.read_csv(file, iterator=False, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @load.register.value(csv_lazy)
    def load_lazy_csv(self, *args, file, mode, size, **kwargs):
        assert mode == "r"
        return dk.read_csv(file, blocksize=size, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @load.register.value(hdf_eager)
    def load_eager_hdf(self, *args, file, mode, **kwargs): pass
    @load.register.value(hdf_lazy)
    def load_lazy_hdf(self, *args, file, mode, **kwargs): pass

    @save.register.value(csv_eager)
    def save_eager_csv(self, dataframe, *args, file, mode, **kwargs):
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @save.register.value(csv_lazy)
    def save_lazy_csv(self, dataframe, *args, file, mode, **kwargs):
        parameters = dict(compute=True, single_file=True, header_first_partition_only=True)
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w", **parameters)

    @save.register.value(hdf_eager)
    def save_eager_hdf(self, dataframe, *args, file, mode, **kwargs): pass
    @save.register.value(hdf_lazy)
    def save_lazy_hdf(self, dataframe, *args, file, mode, **kwargs): pass

    @staticmethod
    def empty(dataframe): return bool(dataframe.empty)

    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates



