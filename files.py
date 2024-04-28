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
from itertools import chain
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.pipelines import Producer, Consumer
from support.processes import Reader, Writer
from support.dispatchers import kwargsdispatcher
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "Files", "FileQuery", "FileTyping", "FileTiming"]
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


class Loader(Reader, Producer):
    def __init__(self, *args, source={}, **kwargs):
        assert isinstance(source, (File, dict))
        assert all([isinstance(file, File) for file in source.keys()])
        super().__init__(*args, source=source, **kwargs)

    def execute(self, *args, **kwargs):
        file = list(self.source.keys())[0]
        for variable, query, extension in file.directory():
            contents = self.read(*args, query=query, **kwargs)
            contents = {variable: content for variable, content in contents.items() if content is not None}
            yield {variable: query} | contents

    def read(self, *args, query, **kwargs):
        contents = {file.variable: file.read(*args, query=query, mode=mode, **kwargs) for file, mode in self.source.items()}
        return contents


class Saver(Writer, Consumer):
    def __init__(self, *args, destination, **kwargs):
        assert isinstance(destination, (File, dict))
        assert all([isinstance(file, File) for file in destination.keys()])
        super().__init__(*args, destination=destination, **kwargs)

    def execute(self, contents, *args, **kwargs):
        contents = {variable: content for variable, content in contents.items() if content is not None}
        self.write(contents, *args, **kwargs)

    def write(self, contents, *args, **kwargs):
        for file, mode in self.destination.items():
            query = contents[str(file.query)]
            content = contents[str(file.variable)]
            file.write(content, *args, query=query, mode=mode, **kwargs)


class FileQuery(ntuple("Query", "variable tostring fromstring")):
    def __str__(self): return str(self.variable)


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.Variable = kwargs.get("variable", getattr(cls, "Variable", None))
        cls.Type = kwargs.get("type", getattr(cls, "Type", None))

    def __call__(cls, *args, **kwargs):
        assert cls.Variable is not None
        assert cls.Type is not None
        parameters = {"variable": cls.Variable, "mutex": FileLock()}
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __hash__(self): return hash(str(self.variable))
    def __init__(self, *args, variable, query, repository, mutex, typing, timing, **kwargs):
        if self.missing(repository):
            os.mkdir(repository)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__repository = repository
        self.__variable = variable
        self.__timing = timing
        self.__typing = typing
        self.__query = query
        self.__mutex = mutex

    def read(self, *args, query, mode, **kwargs):
        file = self.file(query)
        if self.missing(file):
            return
        with self.mutex[file]:
            parameters = dict(file=file, mode=mode, method=self.method)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, query, mode, **kwargs):
        assert content is not None
        file = self.file(query)
        if self.empty(content):
            return
        with self.mutex[file]:
            parameters = dict(file=file, mode=mode, method=self.method)
            self.save(content, *args, **parameters, **kwargs)
        __logger__.info("Saved: {}".format(str(file)))

    def directory(self):
        variable = str(self.query.variable)
        files = list(os.listdir(self.repository))
        files = list(map(lambda x: str(x).split("."), files))
        querys = [(self.query.fromstring(file), extension) for (file, extension) in files]
        for query, extension in querys:
            yield variable, query, extension

    @staticmethod
    @abstractmethod
    def empty(content): pass
    @staticmethod
    def missing(file): return not os.path.exists(file)

    @property
    def method(self): return FileMethod(self.typing, self.timing)
    def file(self, query):
        filename = self.query.tostring(query)
        file = ".".join([filename, str(self.typing.name).lower()])
        return os.path.join(self.repository, file)

    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass

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
    def name(self): return self.__name


class DataframeHeader(ntuple("Header", "index columns")):
    def __iter__(self): return iter([(key, value) for key, value in chain(self.index.items(), self.columns.items())])

class DataframeFile(File, type=pd.DataFrame):
    def __init_subclass__(cls, *args, index, columns, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        assert isinstance(index, dict) and isinstance(columns, dict)
        cls.__header__ = DataframeHeader(index, columns)

    def __init__(self, *args, duplicates=True, **kwargs):
        super().__init__(*args, **kwargs)
        header = self.__class__.__header__
        self.__types = {key: value for (key, value) in iter(header) if not any([value is str, value is np.datetime64])}
        self.__dates = [key for (key, value) in iter(header) if value is np.datetime64]
        self.__duplicates = duplicates
        self.__header = header

    def read(self, *args, **kwargs):
        dataframe = super().read(*args, **kwargs)
        if self.duplicates:
            index = list(self.header.index)
            dataframe = dataframe.drop_duplicates(index, keep="first", inplace=False, ignore_index=True)
        return dataframe

    def write(self, dataframe, *args, **kwargs):
        if self.duplicates:
            index = list(self.header.index)
            dataframe = dataframe.drop_duplicates(index, keep="first", inplace=False, ignore_index=True)
        super().write(dataframe, *args, **kwargs)

    @staticmethod
    def empty(content): return bool(content.empty)

    @kwargsdispatcher("method")
    def load(self, *args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())
    @kwargsdispatcher("method")
    def save(self, content, args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())

    @load.register.value(csv_eager)
    def load_eager_csv(self, *args, file, mode, **kwargs):
        return pd.read_csv(file, iterator=False, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @load.register.value(csv_lazy)
    def load_lazy_csv(self, *args, file, mode, size, **kwargs):
        return dk.read_csv(file, blocksize=size, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @load.register.value(hdf_eager)
    def load_eager_hdf(self, *args, file, mode, **kwargs): pass
    @load.register.value(hdf_lazy)
    def load_lazy_hdf(self, *args, file, mode, **kwargs): pass

    @save.register.value(csv_eager)
    def save_eager_csv(self, content, *args, file, mode, **kwargs):
        content.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @save.register.value(csv_lazy)
    def save_lazy_csv(self, content, *args, file, mode, **kwargs):
        parameters = dict(compute=True, single_file=True, header_first_partition_only=True)
        content.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w", **parameters)

    @save.register.value(hdf_eager)
    def save_eager_hdf(self, content, *args, file, mode, **kwargs): pass
    @save.register.value(hdf_lazy)
    def save_lazy_hdf(self, content, *args, file, mode, **kwargs): pass

    @property
    def duplicates(self): return self.__duplicates
    @property
    def header(self): return self.__header
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates


class Files:
    Dataframe = DataframeFile



