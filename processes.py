# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Process Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Process", "Calculator", "Downloader", "Reader", "Writer"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Process(ABC):
    @typedispatcher
    def size(self, content): raise TypeError(type(content).__name__)
    @size.register(xr.DataArray)
    def size_dataarray(self, dataarray): return np.count_nonzero(~np.isnan(dataarray.values))
    @size.register(pd.Series)
    def size_series(self, series): return len(series.dropna(how="all", inplace=False).index)

    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)


class Calculator(Process, ABC):
    pass


class Downloader(Process, ABC):
    def __init_subclass__(cls, *args, pages={}, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.Pages = {key: value for key, value in pages.items()}

    def __init__(self, *args, feed, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__pages = {key: page(*args, feed=feed, **kwargs) for key, page in self.Pages.items()}

    @property
    def pages(self): return self.__pages


class Reader(Process, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    @abstractmethod
    def read(self, *args, **kwargs): pass
    @property
    def source(self): return self.__source


class Writer(Process, ABC):
    def __init__(self, *args, destination, **kwargs):
        super().__init__(*args, **kwargs)
        self.__destination = destination

    @abstractmethod
    def write(self, *args, **kwargs): pass
    @property
    def destination(self): return self.__destination



