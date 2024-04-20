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

from support.dispatchers import kwargsdispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Archive", "Files", "FileTyping", "FileTiming"]
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


class FileMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.Variable = kwargs.get("variable", getattr(cls, "Variable", None))
        cls.Type = kwargs.get("type", getattr(cls, "Type", None))

    def __call__(cls, *args, **kwargs):
        assert cls.Variable is not None
        assert cls.Type is not None
        parameters = dict(variable=cls.Variable)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, *args, variable, typing, timing, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__variable = variable
        self.__timing = timing
        self.__typing = typing

    def read(self, *args, folder=None, **kwargs):
        file = os.path.join(folder, self.filename) if folder is not None else self.filename
        method = FileMethod(self.typing, self.timing)
        if not os.path.exists(file):
            return
        content = self.load(*args, file=file, method=method, **kwargs)
        if self.empty(content):
            return
        return content

    def write(self, content, *args, folder=None, **kwargs):
        if self.empty(content):
            return
        file = os.path.join(folder, self.filename) if folder is not None else self.filename
        method = FileMethod(self.typing, self.timing)
        self.save(content, *args, file=file, method=method, **kwargs)
        __logger__.info("Saved: {}".format(str(file)))

    @staticmethod
    @abstractmethod
    def empty(content): pass
    @abstractmethod
    def load(self, *args, file, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, **kwargs): pass

    @property
    def filename(self): return ".".join([self.variable, str(self.typing.name).lower()])
    @property
    def variable(self): return self.__variable
    @property
    def timing(self): return self.__timing
    @property
    def typing(self): return self.__typing
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

    def read(self, *args, **kwargs):
        dataframe = super().read(*args, **kwargs)
        if self.duplicates:
            index = list(self.header.index)
            dataframe = dataframe.drop_duplicates(index, keep="first", inplace=False, ignore_index=True)
        return dataframe

    def write(self, dataframe, *args, folder=None, **kwargs):
        if self.duplicates:
            index = list(self.header.index)
            dataframe = dataframe.drop_duplicates(index, keep="first", inplace=False, ignore_index=True)
        super().write(dataframe, *args, **kwargs)

    @kwargsdispatcher("method")
    def load(self, *args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())
    @kwargsdispatcher("method")
    def save(self, content, args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())
    @staticmethod
    def empty(content): return bool(content.empty)

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


class ArchiveMeta(ABCMeta):
    locks = {}

    def __call__(cls, *args, repository, **kwargs):
        lock = ArchiveMeta.locks.get(str(repository),  multiprocessing.RLock())
        ArchiveMeta.locks[str(repository)] = lock
        instance = super(ArchiveMeta, cls).__call__(*args, repository=repository, lock=lock, **kwargs)
        return instance


class Archive(ABC, metaclass=ArchiveMeta):
    def __repr__(self):
        files = chain(self.load.keys(), self.save.keys())
        return f"{self.name}[{', '.join([variable for variable in files])}]"

    def __init__(self, *args, repository, load=[], save=[], lock, **kwargs):
        assert isinstance(load, list) and isinstance(save, list)
        assert all([isinstance(file, File) for file in load])
        assert all([isinstance(file, File) for file in save])
        if not os.path.exists(repository):
            os.makedirs(repository)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__load = {str(file.variable): file for file in load}
        self.__save = {str(file.variable): file for file in save}
        self.__repository = repository
        self.__mutex = lock

    def read(self, *args, folder=None, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        if not os.path.exists(folder):
            return {}
        with self.mutex:
            contents = {variable: file.read(*args, folder=folder, **kwargs) for variable, file in self.load.items()}
            contents = {variable: content for variable, content in contents.items() if content is not None}
            return contents

    def write(self, contents, *args, folder=None, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        if not os.path.exists(folder):
            os.makedirs(folder)
        with self.mutex:
            contents = {variable: (file, contents.get(variable, None)) for variable, file in self.save.items()}
            contents = {variable: (file, content) for variable, (file, content) in contents.items() if content is not None}
            for variable, (file, content) in contents.items():
                file.write(content, *args, folder=folder, **kwargs)

    @property
    def directory(self): return os.listdir(self.repository)
    @property
    def empty(self): return not bool(os.listdir(self.repository))
    @property
    def size(self): return len(os.listdir(self.repository))

    @property
    def repository(self): return self.__repository
    @property
    def mutex(self): return self.__mutex
    @property
    def load(self): return self.__load
    @property
    def save(self): return self.__save
    @property
    def name(self): return self.__name


class Files:
    Dataframe = DataframeFile



