# -*- coding: utf-8 -*-
"""
Created on Weds Aug 5 2026
@name:   File Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeFile"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class DataframeFile(object):
    def __init__(self, *args, header, formatters, parsers, **kwargs):
        assert isinstance(header, dict)
        assert isinstance(formatters, dict)
        assert isinstance(parsers, dict)
        self.__mutex = multiprocessing.Lock()
        self.__formatters = formatters
        self.__parsers = parsers
        self.__header = header

    def save(self, file, dataframe, mode):
        file.parent.mkdir(exist_ok=True, parents=True)
        columns = list(self.header.keys())
        dataframe = dataframe[columns].copy()
        with self.mutex: dataframe.to_csv(file, mode=mode, float_format="%.3f", index=False)

    def load(self, file):
        if not file.exists():
            mapping = {column: pd.Series(dtype=astype) for column, astype in self.header.items()}
            dataframe = pd.DataFrame(mapping)
            return dataframe
        else: return pd.read_csv(file)

    def formatter(self, dataframe):
        for column, formatter in self.formatters.items():
            dataframe[column] = dataframe[column].apply(formatter)
        for column, astype in self.header.items():
            dataframe[column] = dataframe[column].apply(astype)
        return dataframe

    def parser(self, dataframe):
        for column, astype in self.header.items():
            dataframe[column] = dataframe[column].apply(astype)
        for column, parser in self.parsers.items():
            dataframe[column] = dataframe[column].apply(parser)
        return dataframe

    @property
    def formatters(self): return self.__formatters
    @property
    def parsers(self): return self.__parsers
    @property
    def header(self): return self.__header
    @property
    def mutex(self): return self.__mutex




