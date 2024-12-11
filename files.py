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
from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Loader", "Saver", "File"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class FileLock(dict, metaclass=SingletonMeta):
    def __getitem__(self, file):
        self[file] = self.get(file, multiprocessing.RLock())
        return super().__getitem__(file)


class FileMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))

    def __call__(cls, *args, **kwargs):
        parameters = cls.parameters | dict(mutex=FileLock(), folder=cls.variable)
        instance = super(FileMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    def load(cls, *args, file, mode, **kwargs):
        assert mode == "r"
        parameters = dict(infer_datetime_format=False, parse_dates=list(cls.dates.keys()), date_format=cls.dates, dtype=cls.types, converters=cls.parsers)
        dataframe = pd.read_csv(file, iterator=False, index_col=None, header=0, **parameters)
        return dataframe

    def save(cls, dataframe, args, file, mode, **kwargs):
        dataframe = dataframe.copy()
        for column, formatter in cls.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, dateformat in cls.dates.items():
            dataframe[column] = dataframe[column].dt.strftime(dateformat)
        dataframe.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @property
    def variable(cls): return cls.__variable__

    @property
    @abstractmethod
    def formatters(cls): pass
    @property
    @abstractmethod
    def parsers(cls): pass
    @property
    @abstractmethod
    def types(cls): pass
    @property
    @abstractmethod
    def dates(cls): pass


class File(Logging, metaclass=FileMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __new__(cls, *args, repository, **kwargs):
        instance = super().__new__(cls)
        if not os.path.exists(repository):
            os.mkdir(repository)
        return instance

    def __bool__(self): return bool(os.listdir(os.path.join(self.repository, str(self.folder))))
    def __len__(self): return len(os.listdir(os.path.join(self.repository, str(self.folder))))
    def __repr__(self): return f"{self.name}[{len(self):.0f}]"

    def __init__(self, *args, filetiming, filetype, repository, folder, mutex, **kwargs):
        super().__init__(*args, **kwargs)
        self.__repository = repository
        self.__folder = folder
        self.__filetiming = filetiming
        self.__filetype = filetype
        self.__mutex = mutex

    def __iter__(self):
        directory = os.path.join(self.repository, str(self.folder))
        for filename in os.listdir(directory):
            filename = str(filename).split(".")[0]
            yield filename

    def read(self, *args, mode="r", **kwargs):
        directory = os.path.join(self.repository, str(self.folder))
        extension = str(self.filetype.name).lower()
        try: filename = kwargs["filename"]
        except KeyError: filename = self.filename(*args, **kwargs)
        file = os.path.join(directory, ".".join([filename, extension]))
        if not os.path.exists(file): return
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode)
            content = self.load(*args, **parameters, **kwargs)
        return content

    def write(self, content, *args, mode, **kwargs):
        directory = os.path.join(self.repository, str(self.folder))
        extension = str(self.filetype.name).lower()
        try: filename = kwargs["filename"]
        except KeyError: filename = self.filename(*args, **kwargs)
        file = os.path.join(directory, ".".join([filename, extension]))
        with self.mutex[file]:
            parameters = dict(file=str(file), mode=mode)
            self.save(content, *args, **parameters, **kwargs)
        string = f"Saved: {str(file)}"
        self.logger.info(string)

    @staticmethod
    @abstractmethod
    def filename(*args, **kwargs): pass

    @property
    def filetiming(self): return self.__filetiming
    @property
    def filetype(self): return self.__filetype
    @property
    def repository(self): return self.__repository
    @property
    def folder(self): return self.__folder
    @property
    def mutex(self): return self.__mutex


class Process(Logging, Sizing, Emptying, Separating, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        cls.query = kwargs.get("query", getattr(cls, "query", None))

    def __init__(self, *args, file, mode, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()
        self.file = file
        self.mode = mode

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Loader(Process):
    def execute(self, *args, **kwargs):
        if not bool(self.file): return
        for filename in iter(self.file):
            contents = self.file.read(*args, filename=filename, mode=self.mode, **kwargs)
            for group, content in self.separate(contents, *args, keys=list(self.query), **kwargs):
                query = self.query(group)
                size = self.size(content)
                string = f"Loaded: {repr(self)}|{str(query)}[{size:.0f}]"
                self.logger.info(string)
                if self.empty(content): continue
                yield content


class Saver(Process):
    def execute(self, contents, *args, **kwargs):
        if self.empty(contents): return
        for group, content in self.source(contents, *args, keys=list(self.query), **kwargs):
            query = self.query(group)
            self.file.write(content, *args, query=query, mode=self.mode, **kwargs)
            size = self.size(content)
            string = f"Saved: {repr(self)}|{str(query)}[{size:.0f}]"
            self.logger.info(string)





