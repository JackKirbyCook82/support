# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import types
import multiprocessing
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Emptying, Partition, Logging
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Loader", "Saver", "File"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileMeta(ABCMeta):
    def __call__(cls, *args, order=[], **kwargs):
        assert isinstance(order, list) and bool(order)
        split = lambda contents: iter(str(contents).split(" ")) if isinstance(contents, str) else iter(contents)
        formatters = {key: value for keys, value in kwargs.pop("formatters", {}).items() for key in split(keys) if key in order}
        parsers = {key: value for keys, value in kwargs.pop("parsers", {}).items() for key in split(keys) if key in order}
        parameters = dict(formatters=formatters, parsers=parsers, mutex=FileLock())
        parameters["types"] = {key: value for keys, value in kwargs.pop("types", {}).items() for key in split(keys) if key in order}
        parameters["dates"] = {key: value for keys, value in kwargs.pop("dates", {}).items() for key in split(keys) if key in order}
        instance = super(FileMeta, cls).__call__(*args, order=order, **parameters, **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, folder, **kwargs):
        assert repository is not None and folder is not None
        assert os.path.exists(repository)
        instance = super().__new__(cls)
        directory = os.path.join(repository, folder)
        if not os.path.exists(directory):
            os.mkdir(directory)
        return instance

    def __init__(self, *args, repository, folder, order, mutex, **kwargs):
        self.__formatters = kwargs.get("formatters", {})
        self.__parsers = kwargs.get("parsers", {})
        self.__types = kwargs.get("types", {})
        self.__dates = kwargs.get("dates", {})
        self.__repository = repository
        self.__folder = folder
        self.__order = order
        self.__mutex = mutex

    def __bool__(self): return bool(os.listdir(os.path.join(self.repository, self.folder)))
    def __len__(self): return len(os.listdir(os.path.join(self.repository, self.folder)))
    def __iter__(self): return iter(self.directory)

    def read(self, *args, file, mode="r", **kwargs):
        directory = os.path.join(self.repository, self.folder)
        file = os.path.join(directory, file)
        if not os.path.exists(file): return
        with self.mutex[file]: content = self.load(*args, file=file, mode=mode, **kwargs)
        return content

    def write(self, content, *args, file, mode, **kwargs):
        directory = os.path.join(self.repository, self.folder)
        file = os.path.join(directory, file)
        with self.mutex[file]: self.save(content, *args, file=file, mode=mode, **kwargs)
        return file

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
            try: dataframe[column] = dataframe[column].dt.strftime(dateformat)
            except AttributeError: dataframe[column] = dataframe[column].apply(lambda value: value.strftime(dateformat))
        columns = [column for column in list(self.order if bool(self.order) else dataframe.columns) if column in dataframe.columns]
        dataframe[columns].to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @property
    def directory(self):
        directory = os.path.join(self.repository, self.folder)
        return list(os.listdir(directory))

    @property
    def repository(self): return self.__repository
    @property
    def folder(self): return self.__folder
    @property
    def order(self): return self.__order
    @property
    def formatters(self): return self.__formatters
    @property
    def parsers(self): return self.__parsers
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates
    @property
    def mutex(self): return self.__mutex


class Process(Emptying, Logging, ABC):
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


class Loader(Process, Partition, ABC, title="Loaded"):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for file in self.directory(*args, **kwargs):
            dataframe = self.file.read(*args, file=file, mode=self.mode, **kwargs)
            assert isinstance(dataframe, (pd.DataFrame, types.NoneType))
            if self.empty(dataframe): continue
            yield dataframe

    @abstractmethod
    def directory(self, *args, **kwargs): pass


class Saver(Process, Partition, ABC, title="Saved"):
    def execute(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, (pd.DataFrame, types.NoneType))
        if self.empty(dataframe): return
        for file, content in self.categorize(dataframe, *args, **kwargs):
            if self.empty(content): continue
            file = self.file.write(content, *args, file=file, mode=self.mode, **kwargs)
            self.console(file)

    @abstractmethod
    def categorize(self, dataframe, *args, **kwargs): pass



