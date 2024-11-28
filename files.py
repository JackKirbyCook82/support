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
from support.dispatchers import kwargsdispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Loader", "Saver", "File", "FileTypes", "FileTimings"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


FileTypes = Enum("Typing", ["NC", "HDF", "CSV"], start=1)
FileTimings = Enum("Timing", ["EAGER", "LAZY"], start=1)
FileMethod = ntuple("Method", "filetype filetiming")

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


class FileData(ABC, metaclass=RegistryMeta):
    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass

    @classmethod
    def parameters(cls, datatype):
        assert datatype is not None
        keyword = lambda value: value.kind == value.KEYWORD_ONLY
        default = lambda value: value.default if value.default is not value.empty else None
        signature = inspect.signature(cls[datatype].__init__)
        keywords = list(map(keyword, signature.parameters.values()))
        parameters = {keyword.name: default(keyword) for keyword in keywords}
        return parameters


class FileDataframe(FileData, register=pd.DataFrame):
    def __init__(self, *args, formatters={}, parsers={}, dates={}, types={}, **kwargs):
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


class FileMeta(RegistryMeta, ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is FileMeta for base in bases]):
            return super(FileMeta, mcs).__new__(mcs, name, bases, attrs)
        datatype = kwargs.get("datatype", None)
        if datatype is not None:
            bases = tuple(list(bases) + [FileData[datatype]])
            for key in FileData.parameters(datatype): attrs.pop(key)




#        elif not any([FileData not in list(base.__mro__) for base in bases]):
#            datatype = kwargs.get("datatype", None)
#            assert datatype is not None
#            bases = tuple([mixins[datatype]] + list(bases))
#            attrs = {key: value for key, value in attrs.items() if key not in exclude}
#            return super(FileMeta, mcs).__new__(mcs, name, bases, attrs)
#        else:
#            datatypes = set([base.datatype for base in bases if type(base) is FileMeta])
#            assert len(datatypes) == 1 and "datatype" not in kwargs.keys()
#            mixins = {subclass.datatype: subclass for subclass in FileData.__subclasses__()}
#            signature = inspect.signature(mixins[datatypes[0]].__init__)
#            exclude = [value for value in signature.parameters.values() if value.kind == value.KEYWORD_ONLY]
#            attrs = {key: value for key, value in attrs.items() if key not in exclude}
#            return super(FileMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is FileMeta for base in bases]):
            return
        datatype = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        variable = kwargs.get("variable", getattr(cls, "__variable__", None))
        parameters = getattr(cls, "__parameters__", FileData.parameters(datatype))
        parameters = {key: attrs.get(key, value) for key, value in parameters.items()}
        cls.__parameters__ = parameters
        cls.__variable__ = variable
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

    def __init__(self, *args, filetiming, filetype, filename, repository, folder, mutex, **kwargs):
        super().__init__(*args, **kwargs)
        self.__repository = repository
        self.__filetiming = filetiming
        self.__filetype = filetype
        self.__filename = filename
        self.__folder = folder
        self.__mutex = mutex

    def __iter__(self):
        directory = os.path.join(self.repository, str(self.folder))
        for filename in os.listdir(directory):
            filename = str(filename).split(".")[0]
            yield filename

    def read(self, *args, mode="r", **kwargs):
        method = FileMethod(self.filetype, self.filetiming)
        directory = os.path.join(self.repository, str(self.folder))
        extension = str(self.filetype.name).lower()
#        try: filename = kwargs["file"]
#        except KeyError: filename = self.filename(*args, **kwargs)
        file = os.path.join(directory, ".".join([filename, extension]))
        if not os.path.exists(file): return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, mode, **kwargs):
        method = FileMethod(self.filetype, self.filetiming)
        directory = os.path.join(self.repository, str(self.folder))
        extension = str(self.filetype.name).lower()
#        try: filename = kwargs["file"]
#        except KeyError: filename = self.filename(*args, **kwargs)
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
    def folder(self): return self.__folder
    @property
    def mutex(self): return self.__mutex


class Stream(Logging, Sizing, Emptying, Sourcing, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, file, mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = file
        self.mode = mode


class Saver(Stream):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for query, content in self.source(contents, *args, query=self.query, **kwargs):
            self.file.write(content, *args, query=query, mode=self.mode, **kwargs)
            size = self.size(content)
            string = f"Saved: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)


class Loader(Stream):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for filename in iter(self.file):
            contents = self.file.read(*args, file=filename, mode=self.mode, **kwargs)
            for query, content in self.source(contents, *args, query=self.query, **kwargs):
                size = self.size(content)
                string = f"Loaded: {repr(self)}|{str(query)}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(content): continue
                yield content



