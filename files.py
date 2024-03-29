# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Functions/Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from abc import ABC, abstractmethod
from functools import update_wrapper
from collections import OrderedDict as ODict

from support.locks import Locks

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeFile"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class File(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        header = {key: value for key, value in getattr(cls, "__header__", {}).items()}
        update = {key: value for key, value in kwargs.get("header", {}).items()}
        cls.__filename__ = kwargs.get("filename", getattr(cls, "__filename__", None))
        cls.__header__ = header | update

    def __repr__(self): return self.__class__.__name__
    def __bool__(self): return not self.empty if self.table is not None else False
    def __repr__(self): return type(self).__name__
    def __init__(self, *args, repository, timeout=None, **kwargs):
        header = {key: value for key, value in self.__class__.__header__.items()}
        filename = str(self.__class__.__filename__)
        self.__mutex = Locks(timeout=timeout)
        self.__repository = repository
        self.__filename = filename
        self.__header = header

    def load(self, *args, folder=None, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        file = os.path.join(folder, self.filename)
        if not os.path.exists(file):
            return None
        with self.mutex[str(file)]:
            content = load(*args, file=file, type=self.type, **self.parameters, **kwargs)
            return content

    def save(self, content, *args, folder=None, mode, **kwargs):
        assert content is not None
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, self.filename)
        with self.mutex[str(file)]:
            save(content, *args, file=file, mode=mode, **self.parameters, **kwargs)
            __logger__.info("Saved: {}[{}]".format(repr(self), str(file)))

    @property
    def directory(self): return os.listdir(self.repository)
    @property
    def empty(self): return not bool(os.listdir(self.repository))
    @property
    def size(self): return len(os.listdir(self.repository))
    @property
    def name(self): return os.path.splitext(self.filename)[0]

    @property
    @abstractmethod
    def parameters(self): pass

    @property
    def repository(self): return self.__repository
    @property
    def filename(self): return self.__filename
    @property
    def header(self): return self.__header
    @property
    def mutex(self): return self.__mutex


class DataframeFile(File):
    @property
    def parameters(self): return dict(columns=self.columns, types=self.types, dates=self.dates)
    @property
    def types(self): return {key: value for key, value in self.header.items() if not any([value is str, value is np.datetime64])}
    @property
    def dates(self): return [key for key, value in self.header.items() if value is np.datetime64]
    @property
    def columns(self): return list(self.header.keys())


def dispatcher(mainfunction):
    assert callable(mainfunction)
    __method__ = str(mainfunction.__name__)
    __registry__ = ODict()

    def retrieve(filetype, fileext): return __registry__[(filetype, fileext)]
    def update(filetype, fileext, function): __registry__[(filetype, fileext)] = function

    def register(filetype, fileext):
        def decorator(function):
            assert callable(function)
            update(filetype, fileext, function)
            return function
        return decorator

    def save_wrapper(content, *args, file, **kwargs):
        fileext = str(os.path.splitext(file)[-1]).strip(".")
        filetype = type(content)
        try:
            function = retrieve(filetype, fileext)
            return function(content, *args, file=file, **kwargs)
        except IndexError:
            return mainfunction(content, *args, file=file, **kwargs)

    def load_wrapper(*args, file, **kwargs):
        fileext = str(os.path.splitext(file)[-1]).strip(".")
        filetype = kwargs["type"]
        try:
            function = retrieve(filetype, fileext)
            return function(*args, file=file, **kwargs)
        except IndexError:
            return mainfunction(*args, file=file, **kwargs)

    wrappers = dict(save=save_wrapper, load=load_wrapper)
    wrapper = wrappers[mainfunction.__name__]
    wrapper.__method__ = __method__
    wrapper.__registry__ = __registry__
    wrapper.retrieve = retrieve
    wrapper.update = update
    wrapper.register = register
    update_wrapper(wrapper, mainfunction)
    return wrapper


@dispatcher
def save(content, *args, file, **kwargs):
    filetype = type(content).__name__
    fileext = str(os.path.splitext(file)[-1]).strip(".")
    raise ValueError(filetype, fileext)

@save.register(xr.Dataset, "nc")
def save_netcdf(content, *args, file, mode, columns, **kwargs):
    columns = [column for column in columns if column in content.columns]
    xr.Dataset.to_netcdf(content[columns], file, mode=mode, compute=True)

@save.register(pd.DataFrame, "csv")
@save.register(dk.DataFrame, "csv")
def save_csv(content, *args, file, mode, columns, **kwargs):
    parms = dict(index=False, header=not os.path.isfile(file) or mode == "w")
    if isinstance(content, dk.DataFrame):
        update = dict(compute=True, single_file=True, header_first_partition_only=True)
        parms.update(update)
    columns = [column for column in columns if column in content.columns]
    content[columns].to_csv(file, mode=mode, **parms)

@save.register(pd.DataFrame, "hdf")
@save.register(dk.DataFrame, "hdf")
def save_hdf5(self, content, *args, file, mode, columns, **kwargs):
    parms = dict(format="fixed", append=False)
    columns = [column for column in columns if column in content.columns]
    content[columns].to_hdf(file, None, mode=mode, **parms)


@dispatcher
def load(*args, filetype, file, **kwargs):
    filetype = filetype.__name__
    fileext = str(os.path.splitext(file)[-1]).strip(".")
    raise ValueError(filetype, fileext)

@load.register(xr.Dataset, "nc")
def load_netcdf(*args, file, partitions=None, **kwargs):
    return xr.open_dataset(file, chunks=partitions)

@load.register(dk.DataFrame, "csv")
def load_csv_delayed(*args, file, size, columns, types={}, dates=[], **kwargs):
    parms = dict(index_col=None, header=0, dtype=types, parse_dates=dates)
    content = dk.read_csv(file, blocksize=size, **parms)
    columns = [column for column in columns if column in content.columns]
    return content[columns]

@load.register(pd.DataFrame, "csv")
def load_csv_immediate(*args, file, columns, types={}, dates=[], **kwargs):
    parms = dict(index_col=None, header=0, dtype=types, parse_dates=dates)
    content = pd.read_csv(file,  iterator=False, **parms)
    columns = [column for column in columns if column in content.columns]
    return content[columns]

@load.register(pd.DataFrame, "hdf")
def load_hdf5(*args, file, group=None, columns, types={}, dates=[], **kwargs):
    content = pd.read_csv(file, key=group, iterator=False)
    columns = [column for column in columns if column in content.columns]
    return content[columns]









