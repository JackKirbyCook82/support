# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import logging
import multiprocessing
import pandas as pd
import dask.dataframe as dk
from enum import Enum
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple

from support.mixins import Logging, Emptying, Sizing
from support.dispatchers import kwargsdispatcher
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Directory", "Loader", "Saver", "File", "FileTypes", "FileTimings"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


FileTypes = Enum("Typing", ["NC", "HDF", "CSV"], start=1)
FileTimings = Enum("Timing", ["EAGER", "LAZY"], start=1)
FileMethod = ntuple("FileMethod", "filetype filetiming")

csv_eager = FileMethod(FileTypes.CSV, FileTimings.EAGER)
csv_lazy = FileMethod(FileTypes.CSV, FileTimings.LAZY)
hdf_eager = FileMethod(FileTypes.HDF, FileTimings.EAGER)
hdf_lazy = FileMethod(FileTypes.HDF, FileTimings.LAZY)
nc_eager = FileMethod(FileTypes.NC, FileTimings.EAGER)
nc_lazy = FileMethod(FileTypes.NC, FileTimings.LAZY)


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileData(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.datatype = kwargs.get("datatype", getattr(cls, "datatype", None))

    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass


class FileDataframe(FileData, datatype=pd.DataFrame):
    def __init__(self, *args, formatters, parsers, dates, types, **kwargs):
        super().__init__(*args, **kwargs)
        self.__formatters = formatters
        self.__parsers = parsers
        self.__dates = dates
        self.__types = types

    @kwargsdispatcher("method")
    def load(self, *args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())
    @kwargsdispatcher("method")
    def save(self, dataframe, args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())

    @load.register.value(csv_eager)
    def load_eager_csv(self, *args, file, mode="r", **kwargs):
        assert mode == "r"
        parameters = dict(infer_datetime_format=False, parse_dates=list(self.dates.keys()), date_format=self.dates, dtype=self.types, converters=self.parsers)
        dataframe = pd.read_csv(file, iterator=False, index_col=None, header=0, **parameters)
        return dataframe

    @load.register.value(csv_lazy)
    def load_lazy_csv(self, *args, file, mode="r", size, **kwargs):
        assert mode == "r"
        parameters = dict(infer_datetime_format=False, parse_dates=list(self.dates.keys()), date_format=self.dates, dtype=self.types, converters=self.parsers)
        dataframe = dk.read_csv(file, blocksize=size, index_col=None, header=0, **parameters)
        return dataframe

    @save.register.value(csv_eager)
    def save_eager_csv(self, dataframe, *args, file, mode, **kwargs):
        dataframe = dataframe.copy()
        for column, formatter in self.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, dateformat in self.dates.items():
            dataframe[column] = dataframe[column].dt.strftime(dateformat)
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @save.register.value(csv_lazy)
    def save_lazy_csv(self, dataframe, *args, file, mode, **kwargs):
        dataframe = dataframe.copy()
        for column, formatter in self.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, dateformat in self.dates.items():
            dataframe[column] = dataframe[column].dt.strftime(dateformat)
        parameters = dict(compute=True, single_file=True, header_first_partition_only=True)
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w", **parameters)

    @property
    def formatters(self): return self.__formatters
    @property
    def parsers(self): return self.__parsers
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates


class FileMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        mixins = {subclass.datatype: subclass for subclass in FileData.__subclasses__()}
        datatype = kwargs.get("datatype", None)
        if datatype is not None: bases = tuple([mixins[datatype]] + list(bases))
        cls = super(FileMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, **kwargs):
        if not any([type(base) is FileMeta for base in cls.__bases__]):
            cls.__parameters__ = kwargs["parameters"]
        for parameter, default in cls.__parameters__.items():
            existing = getattr(cls, f"__{parameter}__", default)
            updated = kwargs.get(parameter, existing)
            setattr(cls, f"__{parameter}__", updated)

    def __call__(cls, *args, **kwargs):
        parameters = {parameter: getattr(cls, f"__{parameter}__") for parameter in cls.__parameters__}
        instance = super(FileMeta, cls).__call__(*args, **parameters, mutex=FileLock(), **kwargs)
        return instance


class File(Logging, ABC, metaclass=FileMeta, parameters={"variable": None, "formatters": {}, "parsers": {}, "types": {}, "dates": {}}):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __bool__(self): return bool(os.listdir(os.path.join(self.repository, str(self.variable))))
    def __len__(self): return len(os.listdir(os.path.join(self.repository, str(self.variable))))
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"

    def __init__(self, *args, filetiming, filetype, repository, variable, mutex, **kwargs):
        Logging.__init__(self, *args, **kwargs)
        self.__repository = repository
        self.__filetiming = filetiming
        self.__filetype = filetype
        self.__variable = variable
        self.__mutex = mutex

    def read(self, *args, mode="r", **kwargs):
        assert ("file" in kwargs.keys() or "query" in kwargs.keys()) and mode == "r"
        method = FileMethod(self.filetype, self.filetiming)
        directory = os.path.join(self.repository, str(self.variable))
        extension = str(self.filetype.name).lower()
        filename = self.filename(*args, **kwargs)
        file = os.path.join(directory, ".".join([filename, extension]))
        if not os.path.exists(file): return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, mode, **kwargs):
        assert ("file" in kwargs.keys() or "query" in kwargs.keys())
        method = FileMethod(self.filetype, self.filetiming)
        directory = os.path.join(self.repository, str(self.variable))
        extension = str(self.filetype.name).lower()
        filename = self.filename(*args, **kwargs)
        file = os.path.join(directory, ".".join([filename, extension]))
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            self.save(content, *args, **parameters, **kwargs)
        string = f"Saved: {str(file)}"
        self.logger.info(string)

    @property
    def directory(self):
        directory = os.path.join(self.repository, str(self.variable))
        for filename in os.listdir(directory):
            filename = str(filename).split(".")[0]
            parameters = self.parameters(filename=filename)
            assert isinstance(parameters, dict)
            yield parameters, filename

    @staticmethod
    @abstractmethod
    def filename(*args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def parameters(*args, **kwargs): pass

    @property
    def filetiming(self): return self.__filetiming
    @property
    def filetype(self): return self.__filetype
    @property
    def repository(self): return self.__repository
    @property
    def variable(self): return self.__variable
    @property
    def mutex(self): return self.__mutex


class Saver(Logging, Sizing, Emptying):
    def __init__(self, *args, file, mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.__file = file
        self.__mode = mode

    def execute(self, query, content, *args, **kwargs):
        if self.empty(content): return
        self.file.write(content, *args, query=query, mode=self.mode, **kwargs)
        size = self.size(content)
        string = f"Saved: {repr(self)}|{str(query)}[{size:.0f}]"
        self.logger.info(string)

    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Loader(Logging, Sizing, Emptying):
    def __init__(self, *args, query, file, mode="r", **kwargs):
        super().__init__(*args, **kwargs)
        self.__query = query
        self.__file = file
        self.__mode = mode

    def execute(self, query, *args, **kwargs):
        if not bool(self.file): return
        content = self.file.read(*args, query=query, mode=self.mode, **kwargs)
        size = self.size(content)
        string = f"Loaded: {repr(self)}|{str(query)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(content): return
        return content

    @property
    def query(self): return self.__query
    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Directory(Logging, Sizing, Emptying):
    def __init__(self, *args, query, file, mode="r", **kwargs):
        super().__init__(*args, **kwargs)
        self.__query = query
        self.__file = file
        self.__mode = mode

    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for parameters, filename in self.file.directory:
            query = self.query(parameters)
            content = self.file.read(*args, query=query, mode=self.mode, **kwargs)
            size = self.size(content)
            string = f"Loaded: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(content): continue
            yield query, content

    @property
    def query(self): return self.__query
    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode





