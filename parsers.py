# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Parsing Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod

from support.pipelines import Processor
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Parser", "Header"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Parser(Processor, title="Parsed"):
    def __init__(self, *args, header, **kwargs):
        self.__header = header

    def execute(self, *args, **kwargs):
        pass

    @property
    def header(self): return self.__header


class AxesMeta(RegistryMeta):
    def __init__(cls, *args, datatype=None, **kwargs):
        super(AxesMeta, cls).__init__(*args, register=datatype, **kwargs)


class Axes(ABC, metaclass=AxesMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    @abstractmethod
    def parse(self, content, *args, **kwargs): pass


class Dataframe(Axes, datatype=pd.DataFrame):
    def __init__(self, *args, index, columns, duplicates=True, **kwargs):
        super().__init__(*args, **kwargs)
        assert not set(index) & set(columns)
        self.__duplicates = duplicates
        self.__columns = columns
        self.__index = index

    def parse(self, dataframe, *args, **kwargs):
        index = [value for value in self.index if value in dataframe.columns]
        columns = [value for value in self.columns if value in dataframe.columns]
        dataframe = dataframe.drop_duplicates(index, inplace=False) if not self.duplicates else dataframe
        dataframe = dataframe.set_index(index, drop=True, inplace=False)
        dataframe = dataframe[columns]
        return dataframe

    @property
    def duplicates(self): return self.__duplicates
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class HeaderMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is HeaderMeta for base in cls.__bases__]):
            return
        cls.__datatype__ = kwargs.get("datatype", getattr(cls, "__datatype__", None))
        cls.__variable__ = kwargs.get("variable", getattr(cls, "__variable__", None))
        cls.__axes__ = kwargs.get("axes", getattr(cls, "__axes__", None))

    def __call__(cls, *args, **kwargs):
        assert cls.__variable__ is not None
        assert cls.__datatype__ is not None
        assert cls.__axes__ is not None
        axes = Axes[cls.__datatype__](*args, **cls.__axes__, **kwargs)
        instance = super(HeaderMeta, cls).__call__(*args, axes=axes, **kwargs)
        return instance


class Header(Processor, metaclass=HeaderMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, axes, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variable = self.__class__.__variable__
        self.__axes = axes

    def execute(self, contents, *args, **kwargs):
        contents = self.axes.parse(contents, *args, **kwargs)
        return contents

    @property
    def duplicates(self): return self.__duplicates
    @property
    def variable(self): return self.__variable
    @property
    def axes(self): return self.__axes



