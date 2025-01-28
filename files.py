# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import types
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
        attributes = dict(getattr(cls, "__attributes__", {}))
        attributes["formatters"] = kwargs.get("formatters", attributes.get("formatters", {}))
        attributes["parsers"] = kwargs.get("parsers", attributes.get("parsers", {}))
        attributes["types"] = kwargs.get("types", attributes.get("types", {}))
        attributes["dates"] = kwargs.get("dates", attributes.get("dates", {}))
        attributes["order"] = kwargs.get("order", attributes.get("order", {}))
        cls.__attributes__ = attributes

    def __call__(cls, *args, **kwargs):
        parameters = dict(mutex=FileLock()) | dict(cls.attributes)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def attributes(cls): return cls.__attributes__


class File(object, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __init__(self, *args, repository, folder, mutex, order, **kwargs):
        super().__init__(*args, **kwargs)
        self.__formatters = kwargs.get("formatters", {})
        self.__parsers = kwargs.get("parsers", {})
        self.__types = kwargs.get("types", {})
        self.__dates = kwargs.get("dates", {})
        self.__repository = repository
        self.__folder = folder
        self.__mutex = mutex
        self.__order = order

    def __bool__(self): return bool(os.listdir(os.path.join(self.repository, self.folder)))
    def __len__(self): return len(os.listdir(os.path.join(self.repository, self.folder)))
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"

    def __iter__(self):
        directory = os.path.join(self.repository, self.folder)
        for file in os.listdir(directory): yield str(file).split(".")[0]

    def read(self, *args, file, mode="r", **kwargs):
        directory = os.path.join(self.repository, self.folder)
        file = os.path.join(directory, ".".join([file, "csv"]))
        if not os.path.exists(file): return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, file, mode, **kwargs):
        directory = os.path.join(self.repository, self.folder)
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
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates
    @property
    def order(self): return self.__order
    @property
    def mutex(self): return self.__mutex


class Process(Sizing, Emptying, Partition, ABC):
    def __init__(self, *args, file, mode, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.__file = file
        self.__mode = mode

    @staticmethod
    def filename(query): return "_".join(str(query).split("|"))
    @staticmethod
    def queryname(file): return "|".join(str(file).split("_"))

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Directory(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for file in iter(self.file):
            queryname = self.queryname(file)
            query = type(self).query(queryname)
            yield query


class Loader(Process, ABC, title="Loaded"):
    def execute(self, query, *args, **kwargs):
        if query is None: return
        if not bool(self.file): return
        file = self.filename(query)
        dataframes = self.file.read(*args, file=file, mode=self.mode, **kwargs)
        for query, dataframe in self.partition(dataframes):
            size = self.size(dataframe)
            string = f"{str(query)}[{size:.0f}]"
            self.console(string)
            if self.empty(dataframe): continue
            yield dataframe


class Saver(Process, ABC, title="Saved"):
    def execute(self, dataframes, *args, **kwargs):
        assert isinstance(dataframes, (pd.DataFrame, types.NoneType))
        if self.empty(dataframes): return
        for query, dataframe in self.partition(dataframes):
            file = self.filename(query)
            self.file.write(dataframe, *args, file=file, mode=self.mode, **kwargs)
            size = self.size(dataframe)
            string = f"{str(query)}[{size:.0f}]"
            self.console(string)


