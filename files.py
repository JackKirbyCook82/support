# -*- coding: utf-8 -*-
"""
Created on Weds Aug 5 2026
@name:   File Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
from dataclasses import dataclass

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


class FileMeta(type):
    locking = {}

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        header = kwargs.get("header", getattr(cls, "__header__", None))
        cls.__header__ = header

    def __call__(cls, *args, file, **kwargs):
        mutex = FileMeta.locking.get(file, multiprocessing.Lock())
        instance = super().__call__(*args, file=file, mutex=mutex, **kwargs)
        return instance

    @property
    def header(cls): return cls.__header__


class File(object, metaclass=FileMeta):
    def __init__(self, *args, file, mutex, **kwargs):
        assert isinstance(type(self).header, Header)
        self.__mutex = mutex
        self.__file = file

    def save(self, dataframe, mode):
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(mode, str) and mode in ("w", "a")
        self.file.parent.mkdir(exist_ok=True, parents=True)
        dataframe = dataframe[type(self).header.columns].copy()
        for column, formatter in type(self).header.formatting.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, astype in type(self).header.typing.items():
            dataframe[column] = dataframe[column].apply(astype)
        with self.mutex:
            dataframe.to_csv(self.file, mode=mode, float_format="%.3f", index=False)

    def load(self, mode="r"):
        assert isinstance(mode, str) and mode == "r"
        if not self.file.exists():
            mapping = type(self).header.typing.items()
            mapping = {column: pd.Series(dtype=astype) for column, astype in mapping}
            dataframe = pd.DataFrame(mapping)
            return dataframe
        else:
            with self.mutex:
                dataframe = pd.read_csv(self.file)
            for column, astype in type(self).header.typing.items():
                dataframe[column] = dataframe[column].apply(astype)
            for column, parser in type(self).header.parsers.items():
                dataframe[column] = dataframe[column].apply(parser)
            return dataframe

    @property
    def mutex(self): return self.__mutex
    @property
    def file(self): return self.__file



