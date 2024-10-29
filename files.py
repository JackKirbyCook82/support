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

from support.mixins import Function, Generator, Logging, Emptying, Sizing
from support.dispatchers import kwargsdispatcher
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Loader", "Saver", "File", "FileTypes", "FileTimings"]
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
        for parameter in cls.__parameters__:
            existing = getattr(cls, f"__{parameter}__", None)
            updated = kwargs.get(parameter, existing)
            setattr(cls, f"__{parameter}__", updated)

    def __call__(cls, *args, **kwargs):
        parameters = {parameter: getattr(cls, f"__{parameter}__") for parameter in cls.__parameters__}
        parameters = parameters | dict(mutex=FileLock(), name=cls.__name__)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class File(Logging, ABC, metaclass=FileMeta, parameters=["variable", "filename", "formatters", "parsers", "types", "dates"]):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __len__(self): return len(os.listdir(os.path.join(self.repository, str(self.variable))))
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"

    def __init__(self, *args, filetype, filetiming, filename, repository, variable, mutex, name, **kwargs):
        Logging.__init__(self, *args, **kwargs)
        self.__repository = repository
        self.__filetiming = filetiming
        self.__filename = filename
        self.__filetype = filetype
        self.__variable = variable
        self.__mutex = mutex
        self.__name = name

    def directory(self, *args, **kwargs):
        directory = os.path.join(self.repository, str(self.variable))
        for file in os.listdir(directory):
            yield str(file).split(".")[0]

    def read(self, *args, query, mode="r", **kwargs):
        method = FileMethod(self.filetype, self.filetiming)
        directory = os.path.join(self.repository, str(self.variable))
        extension = str(self.filetype.name).lower()
        filename = self.filename(query)
        file = os.path.join(directory, ".".join([filename, extension]))
        if not os.path.exists(file): return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, query, mode, **kwargs):
        method = FileMethod(self.filetype, self.filetiming)
        directory = os.path.join(self.repository, str(self.variable))
        extension = str(self.filetype.name).lower()
        filename = self.filename(query)
        file = os.path.join(directory, ".".join([filename, extension]))
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            self.save(content, *args, **parameters, **kwargs)
        string = f"Saved: {str(file)}"
        self.logger.info(string)

    @property
    def filetiming(self): return self.__filetiming
    @property
    def filetype(self): return self.__filetype
    @property
    def filename(self): return self.__filename
    @property
    def repository(self): return self.__repository
    @property
    def variable(self): return self.__variable
    @property
    def mutex(self): return self.__mutex
    @property
    def name(self): return self.__name


class Saver(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, file, mode, **kwargs):
        assert not inspect.isgeneratorfunction(self.write)
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__file = file
        self.__mode = mode

    def write(self, *args, **kwargs): self.file.write(*args, **kwargs)
    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        query, content = source
        if self.empty(content): return
        self.write(content, *args, query=query, mode=self.mode, **kwargs)
        size = self.size(content)
        string = f"Saved: {repr(self)}|{str(query)}[{size:.0f}]"
        self.logger.info(string)

    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Loader(Generator, Logging, Sizing, Emptying):
    def __init__(self, *args, file, query, mode="r", **kwargs):
        assert not inspect.isgeneratorfunction(self.read)
        Generator.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__file = file
        self.__mode = mode

    def source(self, *args, **kwargs):
        for file in self.file.directory(*args, **kwargs):
            values = str(file).split("_")
            yield self.query(values)

    def read(self, *args, **kwargs): return self.file.read(*args, **kwargs)
    def execute(self, *args, **kwargs):
        for query in self.source(*args, **kwargs):
            content = self.read(*args, query=query, mode=self.mode, **kwargs)
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











