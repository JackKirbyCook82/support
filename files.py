# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import types
import multiprocessing
import regex as re
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Loader", "Saver", "Directory", "File"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileMeta(ABCMeta):
    def __repr__(cls): return str(cls.__name__)
    def __init__(cls, *args, **kwargs):
        super(FileMeta, cls).__init__(*args, **kwargs)
        formatters = kwargs.get("formatters", {})
        generator = lambda keys: iter(str(keys).split(" ")) if isinstance(keys, str) else iter(keys)
        formatters = {key: value for keys, value in formatters.items() for key in generator(keys)}
        assert all([callable(value) for value in formatters.values()])
        attributes = dict(getattr(cls, "__attributes__", {}))
        attributes["formatters"] = attributes.get("formatters", {}) | formatters
        attributes["parsers"] = attributes.get("parsers", {}) | kwargs.get("parsers", {})
        attributes["types"] = attributes.get("types", {}) | kwargs.get("types", {})
        attributes["dates"] = attributes.get("dates", {}) | kwargs.get("dates", {})
        cls.__order__ = kwargs.get("order", getattr(cls, "__order__", []))
        cls.__attributes__ = attributes

    def __getitem__(cls, attribute): return cls.attributes[attribute]
    def __setitem__(cls, attribute, value): cls.attributes[attribute] = value

    def __add__(cls, other):
        primary = list(re.findall("[A-Z][^A-Z]*", repr(cls).replace("File", "")))
        secondary = list(re.findall("[A-Z][^A-Z]*", repr(cls).replace("File", "")))
        titles = list(primary) + [title for title in secondary if title not in primary]
        title = f"{''.join(titles)}File"
        attributes = list(cls.attributes)
        attributes = {attribute: cls[attribute] | other[attribute] for attribute in attributes}
        order = cls.order + [value for value in list(other.order) if value not in cls.order]
        return type(title, (File,), {}, order=order, **attributes)

    def __call__(cls, *args, **kwargs):
        attributes = {attribute: {column: value for column, value in mapping.items() if column in cls.order} for attribute, mapping in cls.attributes.items()}
        parameters = dict(mutex=FileLock(), header=cls.order) | dict(attributes)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def attributes(cls): return cls.__attributes__
    @property
    def order(cls): return cls.__order__


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

    def __init__(self, *args, repository, folder, header, mutex, **kwargs):
        self.__formatters = kwargs.get("formatters", {})
        self.__parsers = kwargs.get("parsers", {})
        self.__types = kwargs.get("types", {})
        self.__dates = kwargs.get("dates", {})
        self.__repository = repository
        self.__folder = folder
        self.__header = header
        self.__mutex = mutex

    def __bool__(self): return bool(os.listdir(os.path.join(self.repository, self.folder)))
    def __len__(self): return len(os.listdir(os.path.join(self.repository, self.folder)))
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"

    def __iter__(self):
        directory = os.path.join(self.repository, self.folder)
        for file in os.listdir(directory):
            file = str(file).split(".")[0]
            yield file

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
        return str(file)

    def load(self, *args, file, mode, **kwargs):
        assert mode == "r" and str(file).split(".")[-1] == "csv"
        parameters = dict(infer_datetime_format=False, parse_dates=list(self.dates.keys()), date_format=self.dates, dtype=self.types, converters=self.parsers)
        dataframe = pd.read_csv(file, iterator=False, index_col=None, header=0, **parameters)
        columns = [column for column in list(self.header if bool(self.header) else dataframe.columns) if column in dataframe.columns]
        return dataframe[columns]

    def save(self, dataframe, *args, file, mode, **kwargs):
        assert str(file).split(".")[-1] == "csv"
        dataframe = dataframe.copy()
        for column, formatter in self.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, dateformat in self.dates.items():
            dataframe[column] = dataframe[column].dt.strftime(dateformat)
        columns = [column for column in list(self.header if bool(self.header) else dataframe.columns) if column in dataframe.columns]
        dataframe[columns].to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @property
    def repository(self): return self.__repository
    @property
    def folder(self): return self.__folder
    @property
    def header(self): return self.__header
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


class Process(Sizing, Emptying, Logging, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, file, mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.__file = file
        self.__mode = mode

    @staticmethod
    def filename(query): return "_".join(str(query).split("|"))
    @staticmethod
    def queryname(file): return "|".join(str(file).split("_"))

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def query(self): return type(self).__query__
    @property
    def file(self): return self.__file
    @property
    def mode(self): return self.__mode


class Directory(Process, ABC):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for file in iter(self.file):
            queryname = self.queryname(file)
            query = self.query(queryname)
            yield query


class Loader(Process, Partition, ABC, title="Loaded"):
    def execute(self, contents, *args, **kwargs):
        contents = list(contents) if isinstance(contents, list) else [contents]
        if not bool(contents): return
        if not bool(self.file): return
        for content in list(contents):
            query = self.query(content)
            file = self.filename(query)
            dataframes = self.file.read(*args, file=file, mode=self.mode, **kwargs)
            for query, dataframe in self.partition(dataframes, by=self.query):
                size = self.size(dataframe)
                self.console(f"{str(query)}[{size:.0f}]")
                if self.empty(dataframe): continue
                yield dataframe


class Saver(Process, Partition, ABC, title="Saved"):
    def execute(self, dataframes, *args, **kwargs):
        assert isinstance(dataframes, (pd.DataFrame, types.NoneType))
        if self.empty(dataframes): return
        for query, dataframe in self.partition(dataframes, by=self.query):
            file = self.filename(query)
            file = self.file.write(dataframe, *args, file=file, mode=self.mode, **kwargs)
            size = self.size(dataframe)
            self.console(f"{str(query)}[{size:.0f}]")
            self.console(file)


