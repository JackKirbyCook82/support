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


class Loader(Producer, ABC, title="Loaded"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__function__ = kwargs.get("function", getattr(cls, "__function__", None))

    def __init__(self, *args, files, directory, wait=0, **kwargs):
        assert isinstance(files, dict) and all([isinstance(file, File) for file in files.keys()])
        super().__init__(*args, **kwargs)
        self.__function = self.__class__.__function__
        self.__directory = directory
        self.__files = files
        self.__wait = int(wait)

    def producer(self, *args, **kwargs):
        for filename in iter(self.directory):
            variable = self.function(filename)
            contents = ODict(list(self.read(*args, variable=variable, **kwargs)))
            yield {self.variable: variable} | contents
            time.sleep(self.wait)

    def read(self, *args, **kwargs):
        for file, mode in self.files.items():
            content = file.read(*args, mode=mode, **kwargs)
            if content is None:
                continue
            yield file.variable, content

    @property
    def directory(self): return self.__directory
    @property
    def function(self): return self.__function
    @property
    def files(self): return self.__files
    @property
    def wait(self): return self.__wait


class Saver(Consumer, ABC, title="Saved"):
    def __init__(self, *args, files, **kwargs):
        assert isinstance(files, dict) and all([isinstance(file, File) for file in files.keys()])
        super().__init__(*args, **kwargs)
        self.__files = files

    def consumer(self, contents, *args, **kwargs):
        variable = contents[self.variable]
        self.write(contents, *args, variable=variable, **kwargs)

    def write(self, contents, *args, **kwargs):
        for file, mode in self.files.items():
            content = contents.get(file.variable, None)
            if content is None:
                continue
            file.write(content, *args, mode=mode, **kwargs)

    @property
    def files(self): return self.__files


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileData(ABC, metaclass=RegistryMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    @abstractmethod
    def load(self, *args, file, mode, **kwargs): pass
    @abstractmethod
    def save(self, content, *args, file, mode, **kwargs): pass
    @staticmethod
    @abstractmethod
    def empty(content): pass


class FileDataframe(FileData, register=pd.DataFrame):
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
        cls.__formatters__ = kwargs.get("formatters", getattr(cls, "__formatters__", {}))
        cls.__parsers__ = kwargs.get("parsers", getattr(cls, "__parsers__", {}))
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__filename__ = kwargs.get("filename", getattr(cls, "__filename__", None))
        cls.__types__ = kwargs.get("types", getattr(cls, "__types__", None))
        cls.__dates__ = kwargs.get("dates", getattr(cls, "__dates__", None))

    def __call__(cls, *args, **kwargs):
        parameters = ("datatype", "variable", "filename", "types", "dates", "formatters", "parsers")
        assert all([parameter is not None for parameter in parameters])
        parameters = dict(parsers=cls.__parsers__, formatters=cls.__formatters__, types=cls.__types__, dates=cls.__dates__)
        filedata = FileData[cls.__datatype__](*args, **parameters, **kwargs)
        parameters = dict(variable=cls.__variable__, filename=cls.__filename__, filedata=filedata, mutex=FileLock())
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __repr__(self): return self.name
    def __init__(self, *args, filedata, filename, filetype, filetiming, repository, mutex, variable, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__repository = repository
        self.__filetiming = filetiming
        self.__filename = filename
        self.__filetype = filetype
        self.__filedata = filedata
        self.__filedata = filedata
        self.__variable = variable
        self.__mutex = mutex

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
            content = self.filedata.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, mode, **kwargs):
        method = FileMethod(self.filetype, self.filetiming)
        file = self.file(*args, **kwargs)
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode, method=method)
            self.filedata.save(content, *args, **parameters, **kwargs)
        __logger__.info("Saved: {}".format(str(file)))

    def file(self, *args, variable, **kwargs):
        directory = os.path.join(self.repository, str(self.variable))
        extension = str(self.filetype.name).lower()
        filename = self.filename(variable)
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
    def filedata(self): return self.__filedata
    @property
    def variable(self): return self.__variable
    @property
    def mutex(self): return self.__mutex
    @property
    def name(self): return self.__name



