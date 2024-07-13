# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import time
import logging
import multiprocessing
import pandas as pd
import dask.dataframe as dk
from enum import Enum
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.pipelines import Producer, Consumer
from support.dispatchers import kwargsdispatcher
from support.meta import SingletonMeta, RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "File", "FileTypes", "FileTimings"]
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


class Loader(Producer, title="Loaded"):
    def __init_subclass__(cls, *args, query, function, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__function__ = function
        cls.__query__ = query

    def __init__(self, *args, directory, source={}, wait=0, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(source, dict) and all([isinstance(file, File) for file in source.keys()])
        self.__function = self.__class__.__function__
        self.__query = self.__class__.__query__
        self.__directory = directory
        self.__source = source
        self.__wait = int(wait)

    def execute(self, *args, **kwargs):
        for filename in iter(self.directory):
            value = self.function(filename)
            contents = ODict(list(self.read(*args, query=value, **kwargs)))
            values = {self.query: value}
            yield values | contents
            time.sleep(self.wait)

    def read(self, *args, **kwargs):
        for file, mode in self.source.items():
            content = file.read(*args, mode=mode, **kwargs)
            if content is None:
                continue
            yield file.variable, content

    @property
    def directory(self): return self.__directory
    @property
    def source(self): return self.__source
    @property
    def function(self): return self.__function
    @property
    def query(self): return self.__query
    @property
    def wait(self): return self.__wait


class Saver(Consumer, title="Saved"):
    def __init_subclass__(cls, *args, query, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = query

    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(destination, dict) and all([isinstance(file, File) for file in destination.keys()])
        self.__query = self.__class__.__query__
        self.__destination = destination

    def execute(self, contents, *args, **kwargs):
        value = contents[self.query]
        self.write(contents, *args, query=value, **kwargs)

    def write(self, contents, *args, **kwargs):
        for file, mode in self.destination.items():
            content = contents.get(file.variable, None)
            if content is None:
                continue
            file.write(content, *args, mode=mode, **kwargs)

    @property
    def destination(self): return self.__destination
    @property
    def query(self): return self.__query


class Lock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class DataMeta(RegistryMeta):
    def __init__(cls, *args, datatype=None, **kwargs):
        super(DataMeta, cls).__init__(*args, register=datatype, **kwargs)


class Data(ABC, metaclass=DataMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass
    @staticmethod
    @abstractmethod
    def empty(content): pass


class Dataframe(Data, datatype=pd.DataFrame):
    def __init__(self, *args, formatters, parsers, dates, types, **kwargs):
        self.__formatters = formatters
        self.__parsers = parsers
        self.__dates = dates
        self.__types = types

    @kwargsdispatcher("method")
    def load(self, *args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())
    @kwargsdispatcher("method")
    def save(self, dataframe, args, file, mode, method, **kwargs): raise ValueError(str(method.name).lower())

    @load.register.value(csv_eager)
    def load_eager_csv(self, *args, file, **kwargs):
        parameters = dict(infer_datetime_format=False, parse_dates=list(self.dates.keys()), date_format=self.dates, dtype=self.types, converters=self.parsers)
        dataframe = pd.read_csv(file, iterator=False, index_col=None, header=0, **parameters)
        return dataframe

    @load.register.value(csv_lazy)
    def load_lazy_csv(self, *args, file, size, **kwargs):
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

    @staticmethod
    def empty(dataframe): return bool(dataframe.empty)
    @property
    def formatters(self): return self.__formatters
    @property
    def parsers(self): return self.__parsers
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates


class FileMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is FileMeta for base in cls.__bases__]):
            return
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__filename__ = kwargs.get("filename", getattr(cls, "__filename__", None))
        cls.__header__ = kwargs.get("header", getattr(cls, "__header__", None))
        cls.__types__ = kwargs.get("types", getattr(cls, "__types__", None))
        cls.__dates__ = kwargs.get("dates", getattr(cls, "__dates__", None))
        cls.__formatters__ = kwargs.get("formatters", getattr(cls, "__formatters__", {}))
        cls.__parsers__ = kwargs.get("parsers", getattr(cls, "__parsers__", {}))

    def __call__(cls, *args, **kwargs):
        assert cls.__datatype__ is not None
        assert cls.__variable__ is not None
        assert cls.__filename__ is not None
        assert cls.__header__ is not None
        assert cls.__types__ is not None
        assert cls.__dates__ is not None
        formatters = {key: value for key, value in cls.__formatters__.items() if key in cls.__header__}
        parsers = {key: value for key, value in cls.__parsers__.items() if key in cls.__header__}
        types = {key: value for key, value in cls.__types__.items() if key in cls.__header__}
        dates = {key: value for key, value in cls.__dates__.items() if key in cls.__header__}
        instance = Data[cls.__datatype__](*args, parsers=parsers, formatters=formatters, types=types, dates=dates, **kwargs)
        parameters = dict(variable=cls.__variable__, filename=cls.__filename__, mutex=Lock())
        instance = super(FileMeta, cls).__call__(instance, *args, **parameters, **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return self.name
    def __init__(self, instance, *args, repository, mutex, variable, filename, filetype, filetiming, **kwargs):
        if not os.path.exists(repository):
            os.mkdir(repository)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__repository = repository
        self.__filetiming = filetiming
        self.__filename = filename
        self.__filetype = filetype
        self.__variable = variable
        self.__mutex = mutex
        self.__data = instance

    def __iter__(self):
        directory = os.path.join(self.repository, str(self.variable))
        filenames = os.listdir(directory)
        for filename in filenames:
            filename, extension = str(filename).split(".")
            assert extension == str(self.filetype.name).lower()
            yield filename

    def read(self, *args, mode, **kwargs):
        method = FileMethod(self.filetype, self.filetiming)
        file = self.file(*args, **kwargs)
        if not os.path.exists(file):
            return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            content = self.data.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, mode, **kwargs):
        method = FileMethod(self.filetype, self.filetiming)
        file = self.file(*args, **kwargs)
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            self.data.save(content, *args, **parameters, **kwargs)
        __logger__.info("Saved: {}".format(str(file)))

    def file(self, *args, query, **kwargs):
        directory = os.path.join(self.repository, str(self.variable))
        extension = str(self.filetype.name).lower()
        filename = self.filename(query)
        return os.path.join(directory, ".".join([filename, extension]))

    @property
    def repository(self): return self.__repository
    @property
    def filetiming(self): return self.__filetiming
    @property
    def filename(self): return self.__filename
    @property
    def filetype(self): return self.__filetype
    @property
    def variable(self): return self.__variable
    @property
    def mutex(self): return self.__mutex
    @property
    def data(self): return self.__data
    @property
    def name(self): return self.__name



