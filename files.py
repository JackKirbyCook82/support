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
from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Emptying, Sizing, Partition
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Loader", "Saver", "Directory", "File"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        super(FileMeta, cls).__init__(*args, **kwargs)
        variable = kwargs.get("variable", getattr(cls, "__variable__", None))
        order = kwargs.get("order", getattr(cls, "attributes", {}).get("order", []))
        formatters = kwargs.get("formatters", getattr(cls, "attributes", {}).get("formatters", []))
        parsers = kwargs.get("parsers", getattr(cls, "attributes", {}).get("parsers", []))
        types = kwargs.get("types", getattr(cls, "attributes", {}).get("types", []))
        dates = kwargs.get("dates", getattr(cls, "attributes", {}).get("dates", []))
        attributes = dict(order=order, formatters=formatters, parsers=parsers, types=types, dates=dates)
        cls.__attributes__ = attributes
        cls.__variable__ = variable

    def __call__(cls, *args, **kwargs):
        formatters = {key: cls.attributes["formatters"].get(key, {}) for key in cls.attributes["order"]}
        parsers = {key: cls.attributes["parsers"].get(key, {}) for key in cls.attributes["order"]}
        types = {key: cls.attributes["types"].get(key, {}) for key in cls.attributes["order"]}
        dates = {key: cls.attributes["dates"].get(key, {}) for key in cls.attributes["order"]}
        parameters = dict(mutex=FileLock(), folder=cls.variable, formatters=formatters, parsers=parsers, types=types, dates=dates)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def attributes(cls): return cls.__attributes__
    @property
    def variable(cls): return cls.__variable__


class File(object, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __init__(self, *args, repository, folder, mutex, order, formatters, parsers, types, dates, **kwargs):
        super().__init__(*args, **kwargs)
        self.__repository = repository
        self.__folder = folder
        self.__formatters = formatters
        self.__parsers = parsers
        self.__order = order
        self.__types = types
        self.__dates = dates
        self.__mutex = mutex

    def __bool__(self): return bool(os.listdir(os.path.join(self.repository, str(self.folder))))
    def __len__(self): return len(os.listdir(os.path.join(self.repository, str(self.folder))))
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"

    def __iter__(self):
        directory = os.path.join(self.repository, str(self.folder))
        for file in os.listdir(directory): yield str(file).split(".")[0]

    def read(self, *args, file, mode="r", **kwargs):
        directory = os.path.join(self.repository, str(self.folder))
        file = os.path.join(directory, ".".join([file, "csv"]))
        if not os.path.exists(file): return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, file, mode, **kwargs):
        directory = os.path.join(self.repository, str(self.folder))
        file = os.path.join(directory, ".".join([file, "csv"]))
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode)
            self.save(content, *args, **parameters, **kwargs)
        string = f"Saved: {str(file)}"
        __logger__.info(string)

    def load(self, *args, file, mode, **kwargs):
        assert mode == "r" and str(file).split(".")[-1] == "csv"
        parameters = dict(infer_datetime_format=False, parse_dates=list(self.dates.keys()), date_format=self.dates, dtype=self.types, converters=self.parsers)
        dataframe = pd.read_csv(file, iterator=False, index_col=None, header=0, **parameters)
        columns = [column for column in list(self.order if bool(self.order) else dataframe.columns) if column in dataframe.columns]
        return dataframe[columns]

    def save(self, dataframe, *args, file, mode, **kwargs):
        assert str(file).split(".")[-1] == "csv"
        dataframe = dataframe.copy()
        for column, formatter in self.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, dateformat in self.dates.items():
            dataframe[column] = dataframe[column].dt.strftime(dateformat)
        columns = [column for column in list(self.order if bool(self.order) else dataframe.columns) if column in dataframe.columns]
        dataframe[columns].to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @property
    def repository(self): return self.__repository
    @property
    def folder(self): return self.__folder
    @property
    def formatters(self): return self.__formatters
    @property
    def parsers(self): return self.__parsers
    @property
    def order(self): return self.__order
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates
    @property
    def mutex(self): return self.__mutex


class Process(Sizing, Emptying, ABC):
    def __init__(self, *args, file, mode, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.__file = file
        self.__mode = mode

    @staticmethod
    def filename(query): return str(query).replace("|", "_")
    @staticmethod
    def filevalues(string): return str(string).split("_")

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Directory(Process):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for file in iter(self.file):
            values = self.filevalues(file)
            query = self.query(values)
            yield query


class Loader(Partition, Process):
    def execute(self, query, *args, **kwargs):
        if query is None: return
        if not bool(self.file): return
        file = self.filename(query)
        contents = self.file.read(*args, file=file, mode=self.mode, **kwargs)
        for query, content in self.separate(contents, *args, **kwargs):
            size = self.size(content)
            string = f"Loaded: {repr(self)}|{str(query)}[{size:.0f}]"
            __logger__.info(string)
            if self.empty(content): continue
            yield content


class Saver(Partition, Process):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for query, content in self.segregate(contents, *args, **kwargs):
            file = self.filename(query)
            self.file.write(content, *args, file=file, mode=self.mode, **kwargs)
            size = self.size(content)
            string = f"Saved: {repr(self)}|{str(query)}[{size:.0f}]"
            __logger__.info(string)


