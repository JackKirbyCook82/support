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

from support.mixins import Logging, Emptying, Sizing, Separating
from support.decorators import TypeDispatcher
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
    fields = ("order", "formatters", "parsers", "types", "dates")

    def __init__(cls, *args, **kwargs):
        variable = kwargs.get("variable", getattr(cls, "__variable__", None))
        parameters = {field: getattr(cls, "parameters", {}).get(field, None) for field in type(cls).fields}
        parameters = {key: kwargs.get(key, value) for key, value in parameters.items()}
        cls.__parameters__ = parameters
        cls.__variable__ = variable

    def __call__(cls, *args, **kwargs):
        parameters = dict(mutex=FileLock(), folder=cls.variable, **cls.parameters)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def parameters(cls): return cls.__parameters__
    @property
    def variable(cls): return cls.__variable__


class File(Logging, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __init__(self, *args, repository, folder, mutex, **kwargs):
        super().__init__(*args, **kwargs)
        self.__formatters = kwargs.get("formatters", {})
        self.__parsers = kwargs.get("parsers", {})
        self.__order = kwargs.get("order", [])
        self.__types = kwargs.get("types", {})
        self.__dates = kwargs.get("dates", {})
        self.__repository = repository
        self.__folder = folder
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
        self.logger.info(string)

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
    def repository(self): return self.__repository
    @property
    def folder(self): return self.__folder
    @property
    def mutex(self): return self.__mutex


class Process(Logging, Sizing, Emptying, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, file, mode, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.__file = file
        self.__mode = mode

    @staticmethod
    def filename(query): return str(query).replace("|", "_")
    @property
    def fieldnames(self): return list(self.query)
    @TypeDispatcher(locator=0)
    def queryname(self, parameters): return self.query(parameters)
    @queryname.register(str)
    def string(self, string): return self.query[str(string).replace("_", "|")]

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def fields(self): return list(type(self).__query__)
    @property
    def query(self): return type(self).__query__
    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Directory(Process):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for file in iter(self.file):
            query = self.queryname(file)
            yield query


class Loader(Process, Separating):
    def execute(self, query, *args, **kwargs):
        if query is None: return
        if not bool(self.file): return
        query = self.queryname(query)
        file = self.filename(query)
        contents = self.file.read(*args, file=file, mode=self.mode, **kwargs)
        for parameters, content in self.separate(contents, *args, fields=self.fieldnames, **kwargs):
            query = self.queryname(parameters)
            size = self.size(content)
            string = f"Loaded: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(content): continue
            yield content


class Saver(Process, Separating):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for parameters, content in self.separate(contents, *args, fields=self.fieldnames, **kwargs):
            query = self.queryname(parameters)
            file = self.filename(query)
            self.file.write(content, *args, file=file, mode=self.mode, **kwargs)
            size = self.size(content)
            string = f"Saved: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)


