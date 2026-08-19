# -*- coding: utf-8 -*-
"""
Created on Weds Aug 5 2026
@name:   File Objects
@author: Jack Kirby Cook

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


class File(Logging, ABC):
    locking = {}

    def __init_subclass__(cls, header, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(header, Header)
        cls.__header__ = header

    def __new__(cls, *args, file, **kwargs):
        mutex = File.locking.get(file, multiprocessing.Lock())
        File.locking[file] = mutex
        instance = super().__new__(cls, *args, **kwargs)
        instance.mutex = mutex
        return instance

    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, Path)
        super().__init__(*args, **kwargs)
        self.__mutex = None
        self.__file = file

    def save(self, dataframe, mode):
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(mode, str) and mode in ("w", "a")
        self.file.parent.mkdir(exist_ok=True, parents=True)
        dataframe = dataframe[self.header.columns].copy()
        for column, formatter in self.header.formatting.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, astype in self.header.typing.items():
            dataframe[column] = dataframe[column].apply(astype)
        with self.mutex:
            dataframe.to_csv(self.file, mode=mode, float_format="%.3f", index=False)
        self.results(dataframe, title="Saved")

    def load(self, mode="r"):
        assert isinstance(mode, str) and mode == "r"
        if not self.file.exists():
            mapping = self.header.typing.items()
            mapping = {column: pd.Series(dtype=astype) for column, astype in mapping}
            dataframe = pd.DataFrame(mapping)
            return dataframe
        with self.mutex:
            dataframe = pd.read_csv(self.file)
        for column, astype in self.header.typing.items():
            dataframe[column] = dataframe[column].apply(astype)
        for column, parser in self.header.parsers.items():
            dataframe[column] = dataframe[column].apply(parser)
        self.results(dataframe, title="Loaded")
        return dataframe

    @abstractmethod
    def results(self, dataframe, *args, title, **kwargs): pass

    @property
    def header(self): return type(self).__header__
    @property
    def file(self): return self.__file

    @property
    def mutex(self): return self.__mutex
    @mutex.setter
    def mutex(self, mutex): self.__mutex = mutex




