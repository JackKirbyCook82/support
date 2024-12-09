# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import inspect
import logging
import multiprocessing
import pandas as pd
import dask.dataframe as dk
from enum import Enum
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple

from support.mixins import Logging, Emptying, Sizing, Sourcing
from support.meta import SingletonMeta, RegistryMeta
from support.decorators import ValueDispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Directory", "Loader", "Saver", "Process", "File", "FileTypes", "FileTimings"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


FileTimings = Enum("Timing", ["EAGER", "LAZY"], start=1)
FileTypes = Enum("Typing", ["NC", "HDF", "CSV"], start=1)
FileMethod = ntuple("Method", "filetype filetiming datatype")
EagerTableCSV = FileMethod(FileTimings.EAGER, pd.DataFrame, FileTypes.CSV)
LazyTableCSV = FileMethod(FileTimings.LAZY, pd.DataFrame, FileTypes.CSV)
EagerTableHDF = FileMethod(FileTimings.EAGER, pd.DataFrame, FileTypes.HDF)
LazyTableHDF = FileMethod(FileTimings.LAZY, pd.DataFrame, FileTypes.HDF)
EagerTableNC = FileMethod(FileTimings.EAGER, pd.DataFrame, FileTypes.NC)
LazyTableNC = FileMethod(FileTimings.LAZY, pd.DataFrame, FileTypes.NC)


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileData(ABC, metaclass=RegistryMeta):
    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass

    @classmethod
    def defaults(cls, datatype):
        assert datatype is not None
        keyword = lambda value: value.kind == value.KEYWORD_ONLY
        default = lambda value: value.default if value.default is not value.empty else None
        signature = inspect.signature(cls[datatype].__init__)
        keywords = list(filter(keyword, signature.parameters.values()))
        parameters = {keyword.name: default(keyword) for keyword in keywords}
        return parameters


class FileTableStream(FileData, ABC):
    @property
    @abstractmethod
    def formatters(self): pass
    @property
    @abstractmethod
    def parsers(self): pass
    @property
    @abstractmethod
    def types(self): pass
    @property
    @abstractmethod
    def dates(self): pass


class FileTableLoading(FileTableStream, ABC):
    @ValueDispatcher(locator="method")
    def load(self, *args, file, mode, method, **kwargs): raise ValueError(method)

    @load.register(EagerTableCSV)
    def __eagercsv(self, *args, file, mode="r", **kwargs):
        assert mode == "r"
        parameters = dict(infer_datetime_format=False, parse_dates=list(self.dates.keys()), date_format=self.dates, dtype=self.types, converters=self.parsers)
        dataframe = pd.read_csv(file, iterator=False, index_col=None, header=0, **parameters)
        return dataframe

    @load.register(LazyTableCSV)
    def __lazycsv(self, *args, file, mode="r", size, **kwargs):
        assert mode == "r"
        parameters = dict(infer_datetime_format=False, parse_dates=list(self.dates.keys()), date_format=self.dates, dtype=self.types, converters=self.parsers)
        dataframe = dk.read_csv(file, blocksize=size, index_col=None, header=0, **parameters)
        return dataframe


class FileTableSaving(FileTableStream, ABC):
    @ValueDispatcher(locator="method")
    def save(self, dataframe, args, file, mode, method, **kwargs): raise ValueError(method)

    @save.register(EagerTableCSV)
    def __eagercsv(self, dataframe, *args, file, mode, **kwargs):
        dataframe = dataframe.copy()
        for column, formatter in self.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, dateformat in self.dates.items():
            dataframe[column] = dataframe[column].dt.strftime(dateformat)
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @save.register(LazyTableCSV)
    def __lazycsv(self, dataframe, *args, file, mode, **kwargs):
        dataframe = dataframe.copy()
        for column, formatter in self.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, dateformat in self.dates.items():
            dataframe[column] = dataframe[column].dt.strftime(dateformat)
        parameters = dict(compute=True, single_file=True, header_first_partition_only=True)
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w", **parameters)


class FileTable(FileTableLoading, FileTableSaving, register=pd.DataFrame):
    def __init__(self, *args, formatters={}, parsers={}, dates={}, types={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.__formatters = formatters
        self.__parsers = parsers
        self.__dates = dates
        self.__types = types

    @property
    def formatters(self): return self.__formatters
    @property
    def parsers(self): return self.__parsers
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates


class FileMeta(RegistryMeta, ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is FileMeta for base in bases]):
            return super(FileMeta, mcs).__new__(mcs, name, bases, attrs)
        datatype = kwargs.get("datatype", None)
        if datatype is not None:
            bases = tuple([FileData[datatype]] + list(bases))
            exclude = FileData.defaults(datatype).keys()
            attrs = {key: value for key, value in attrs.items() if key not in exclude}
        return super(FileMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs, *args, datatype=None, variable=None, **kwargs):
        super(FileMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        if not any([type(base) is FileMeta for base in bases]):
            return
        if variable is not None:
            cls.__variable__ = variable
        if datatype is not None:
            parameters = FileData.defaults(datatype)
            parameters = {key: attrs.get(key, value) for key, value in parameters.items()}
            cls.__parameters__ = parameters
            cls.__datatype__ = datatype

    def __call__(cls, *args, **kwargs):
        parameters = cls.parameters | dict(mutex=FileLock(), folder=cls.variable)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def parameters(cls): return cls.__parameters__
    @property
    def variable(cls): return cls.__variable__
    @property
    def datatype(cls): return cls.__datatype__


class File(Logging, ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __bool__(self): return bool(os.listdir(os.path.join(self.repository, str(self.folder))))
    def __len__(self): return len(os.listdir(os.path.join(self.repository, str(self.folder))))
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"

    def __init__(self, *args, filetiming, filetype, repository, folder, mutex, **kwargs):
        super().__init__(*args, **kwargs)
        self.__repository = repository
        self.__filetiming = filetiming
        self.__filetype = filetype
        self.__folder = folder
        self.__mutex = mutex

    def __iter__(self):
        directory = os.path.join(self.repository, str(self.folder))
        for filename in os.listdir(directory):
            filename = str(filename).split(".")[0]
            yield filename

    def read(self, *args, mode="r", **kwargs):
        method = FileMethod(self.filetiming, type(self).datatype, self.filetype)
        directory = os.path.join(self.repository, str(self.folder))
        extension = str(self.filetype.name).lower()
        try: filename = kwargs["filename"]
        except KeyError: filename = self.filename(*args, **kwargs)
        file = os.path.join(directory, ".".join([filename, extension]))
        if not os.path.exists(file): return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, mode, **kwargs):
        method = FileMethod(self.filetiming, type(self).datatype, self.filetype)
        directory = os.path.join(self.repository, str(self.folder))
        extension = str(self.filetype.name).lower()
        try: filename = kwargs["filename"]
        except KeyError: filename = self.filename(*args, **kwargs)
        file = os.path.join(directory, ".".join([filename, extension]))
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            self.save(content, *args, **parameters, **kwargs)
        string = f"Saved: {str(file)}"
        self.logger.info(string)

    @staticmethod
    @abstractmethod
    def filename(*args, **kwargs): pass

    @property
    def filetiming(self): return self.__filetiming
    @property
    def filetype(self): return self.__filetype
    @property
    def repository(self): return self.__repository
    @property
    def folder(self): return self.__folder
    @property
    def mutex(self): return self.__mutex


class Process(Logging, Sizing, Emptying, Sourcing, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        cls.title = kwargs.get("title", getattr(cls, "title", None))
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, file, mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.__file = file
        self.__mode = mode

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Directory(Process, title="Loaded"):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for filename in iter(self.file):
            contents = self.file.read(*args, filename=filename, mode=self.mode, **kwargs)
            for query, content in self.source(contents, *args, query=self.query, **kwargs):
                size = self.size(content)
                string = f"{str(self.title)}: {repr(self)}|{str(query)}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(content): continue
                yield content


class Loader(Process, title="Loaded"):
    def execute(self, query, *args, **kwargs):
        if query is None: return
        query = self.query(query)
        content = self.file.read(*args, query=query, mode=self.mode, **kwargs)
        size = self.size(content)
        string = f"{str(self.title)}: {repr(self)}|{str(query)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(content): return
        return content


class Saver(Process, title="Saved"):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for query, content in self.source(contents, *args, query=self.query, **kwargs):
            self.file.write(content, *args, query=query, mode=self.mode, **kwargs)
            size = self.size(content)
            string = f"{str(self.title)}: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)



