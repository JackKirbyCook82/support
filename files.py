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

from support.pipelines import Producer, Consumer
from support.processes import Reader, Writer
from support.dispatchers import kwargsdispatcher
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Files", "FileTyping", "FileTiming"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


FileTyping = IntEnum("FileTyping", ["NC", "HDF", "CSV"], start=1)
FileTiming = IntEnum("FileTiming", ["EAGER", "LAZY"], start=1)
FileMethod = ntuple("FileMethod", "filetyping filetiming")

csv_eager = FileMethod(FileTyping.CSV, FileTiming.EAGER)
csv_lazy = FileMethod(FileTyping.CSV, FileTiming.LAZY)
hdf_eager = FileMethod(FileTyping.HDF, FileTiming.EAGER)
hdf_lazy = FileMethod(FileTyping.HDF, FileTiming.LAZY)
nc_eager = FileMethod(FileTyping.NC, FileTiming.EAGER)
nc_lazy = FileMethod(FileTyping.NC, FileTiming.LAZY)


class Loader(Reader, Producer):
    def __contains__(self, variable): return variable in self.source.keys()
    def __init__(self, *args, source, repository, query, mode="r", **kwargs):
        assert isinstance(source, (File, list))
        assert callable(query)
        source = [source] if isinstance(source, File) else source
        assert all([isinstance(file, File) for file in source])
        source = {file.variable: file for file in source}
        super().__init__(*args, source=source, **kwargs)
        self.__repository = repository
        self.__query = query
        self.__mode = mode

    def execute(self, *args, **kwargs):
        for folder in self.directory(*args, **kwargs):
            query = self.query(folder, *args, **kwargs)
            contents = self.read(*args, folder=folder, **kwargs)
            yield query | contents

    def read(self, *args, folder, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        contents = {variable: file.read(*args, folder=folder, **kwargs) for variable, file in self.source.items()}
        contents = {variable: content for variable, content in contents.items() if content is not None}
        return contents

    @property
    def repository(self): return self.__repository
    @property
    def query(self): return self.__query
    @property
    def mode(self): return self.__mode


class Saver(Writer, Consumer):
    def __contains__(self, variable): return variable in self.destination.keys()
    def __init__(self, *args, destination, repository, folder, mode, **kwargs):
        assert isinstance(destination, (File, list))
        assert callable(folder)
        destination = [destination] if isinstance(destination, File) else destination
        assert all([isinstance(file, File) for file in destination])
        destination = {file.variable: file for file in destination}
        super().__init__(*args, destination=destination, **kwargs)
        self.__repository = repository
        self.__folder = folder
        self.__mode = mode

    def execute(self, contents, *args, **kwargs):
        assert isinstance(contents, dict)
        folder = self.folder(contents, *args, **kwargs)
        contents = {variable: content for variable, content in contents.items() if variable in self}
        self.write(contents, *args, folder=folder, **kwargs)

    def write(self, contents, *args, folder, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        contents = {variable: content for variable, content in contents.items() if content is not None}
        for variable, content in contents.items():
            self.destination[variable](content, *args, folder=folder, **kwargs)

    @property
    def repository(self): return self.__repository
    @property
    def folder(self): return self.__folder
    @property
    def mode(self): return self.__mode


class FileLock(dict, SingletonMeta):
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
        mutex = FileLock()
        parameters = dict(variable=cls.Variable, mutex=mutex)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, *args, variable, mutex, typing, timing, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__variable = variable
        self.__timing = timing
        self.__typing = typing
        self.__mutex = mutex

    def load(self, *args, folder, **kwargs):
        method = FileMethod(self.typing, self.timing)
        filename = ".".join([self.variable, str(self.typing.name).lower()])
        file = os.path.join(folder, filename)
        if self.missing(file):
            return
        with self.mutex[file]:
            content = self.loader(*args, file=file, method=method, **kwargs)
        return content

    def save(self, content, *args, folder, **kwargs):
        assert content is not None
        if self.empty(content):
            return
        method = FileMethod(self.typing, self.timing)
        filename = ".".join([self.variable, str(self.typing.name).lower()])
        file = os.path.join(folder, filename)
        with self.mutex[file]:
            self.saver(content, *args, file=file, method=method, **kwargs)
        __logger__.info("Saved: {}".format(str(file)))

    @staticmethod
    @abstractmethod
    def empty(content): pass
    @staticmethod
    def missing(file): return not os.path.exists(file)

    @abstractmethod
    def loader(self, *args, file, **kwargs): pass
    @abstractmethod
    def saver(self, content, *args, file, **kwargs): pass

    @property
    def variable(self): return self.__variable
    @property
    def timing(self): return self.__timing
    @property
    def typing(self): return self.__typing
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

    def load(self, *args, **kwargs):
        dataframe = super().load(*args, **kwargs)
        if self.duplicates:
            index = list(self.header.index)
            dataframe = dataframe.drop_duplicates(index, keep="first", inplace=False, ignore_index=True)
        return dataframe

    def save(self, dataframe, *args, **kwargs):
        if self.duplicates:
            index = list(self.header.index)
            dataframe = dataframe.drop_duplicates(index, keep="first", inplace=False, ignore_index=True)
        super().save(dataframe, *args, **kwargs)

    @staticmethod
    def empty(content): return bool(content.empty)

    @kwargsdispatcher("method")
    def loader(self, *args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())
    @kwargsdispatcher("method")
    def saver(self, content, args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())

    @load.register.value(csv_eager)
    def loader_eager_csv(self, *args, file, mode, **kwargs):
        return pd.read_csv(file, iterator=False, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @load.register.value(csv_lazy)
    def loader_lazy_csv(self, *args, file, mode, size, **kwargs):
        return dk.read_csv(file, blocksize=size, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @load.register.value(hdf_eager)
    def loader_eager_hdf(self, *args, file, mode, **kwargs): pass
    @load.register.value(hdf_lazy)
    def loader_lazy_hdf(self, *args, file, mode, **kwargs): pass

    @save.register.value(csv_eager)
    def saver_eager_csv(self, content, *args, file, mode, **kwargs):
        content.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @save.register.value(csv_lazy)
    def saver_lazy_csv(self, content, *args, file, mode, **kwargs):
        parameters = dict(compute=True, single_file=True, header_first_partition_only=True)
        content.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w", **parameters)

    @save.register.value(hdf_eager)
    def saver_eager_hdf(self, content, *args, file, mode, **kwargs): pass
    @save.register.value(hdf_lazy)
    def saver_lazy_hdf(self, content, *args, file, mode, **kwargs): pass

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



