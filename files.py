# -*- coding: utf-8 -*-
"""
Created on Weds Aug 5 2026
@name:   File Objects
@author: Jack Kirby Cook
@file:   support/files.py

"""

import multiprocessing
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["File", "Header"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass
class Header:
    columns: list; typing: dict; formatting: dict; parsers: dict

    def __post_init__(self):
        self.typing = {key: value for key, value in self.typing.items() if key in self.columns}
        self.formatting = {key: value for key, value in self.formatting.items() if key in self.columns}
        self.parsers = {key: value for key, value in self.parsers.items() if key in self.columns}


class FileMeta(type(Logging), type):
    locking = {}

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parameters = getattr(cls, "__parameters__", {})
        header = kwargs.get("header", parameters.get("header", None))
        parameters.update({"header": header})
        cls.__parameters__ = parameters

    def __call__(cls, *args, file, **kwargs):
        mutex = FileMeta.locking.get(file, multiprocessing.Lock())
        FileMeta.locking[file] = mutex
        parameters = dict(file=file, mutex=mutex) | cls.parameters
        instance = super().__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def parameters(cls): return cls.__parameters__


class File(Logging, ABC, metaclass=FileMeta):
    def __init__(self, *args, file, mutex, header, **kwargs):
        assert isinstance(file, Path)
        super().__init__(*args, **kwargs)
        self.__header = header
        self.__mutex = mutex
        self.__file = file

    def save(self, dataframe, mode, columns=None):
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(mode, str) and mode in ("w", "a")
        if dataframe.empty: return
        self.file.parent.mkdir(exist_ok=True, parents=True)
        columns = self.header.columns if columns is None else columns
        dataframe = dataframe[columns].copy()
        for column, formatter in self.header.formatting.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, astype in self.header.typing.items():
            dataframe[column] = dataframe[column].apply(astype)
        with self.mutex:
            dataframe.to_csv(self.file, mode=mode, float_format="%.3f", index=False)
        self.results(dataframe, title="Saved")

    def load(self, mode="r", columns=None):
        assert isinstance(mode, str) and mode == "r"
        if not self.file.exists():
            mapping = self.header.typing.items()
            mapping = {column: pd.Series(dtype=astype) for column, astype in mapping}
            dataframe = pd.DataFrame(mapping)
            return dataframe
        columns = self.header.columns if columns is None else columns
        with self.mutex:
            dataframe = pd.read_csv(self.file)
        for column, astype in self.header.typing.items():
            dataframe[column] = dataframe[column].apply(astype)
        for column, parser in self.header.parsers.items():
            dataframe[column] = dataframe[column].apply(parser)
        dataframe = dataframe[columns]
        self.results(dataframe, title="Loaded")
        return dataframe

    @abstractmethod
    def results(self, dataframe, *args, title, **kwargs): pass

    @property
    def header(self): return type(self).__header__
    @property
    def mutex(self): return self.__mutex
    @property
    def file(self): return self.__file







