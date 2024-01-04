# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Functions/Objects
@author: Jack Kirby Cook

"""

import os.path
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from functools import update_wrapper
from collections import OrderedDict as ODict

from support.locks import Locks
from support.pipelines import Stack

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeFile"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = ""


class File(Stack):
    def __init__(self, *args, repository, capacity=None, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        name = str(self.name).replace("File", "Lock")
        self.__mutex = Locks(name=name, timeout=timeout)
        self.__repository = repository
        self.__capacity = capacity
        self.__timeout = timeout

    def folders(self, *path): return (folder for folder in os.listdir(self.repository, *path) if os.path.isdir(folder))
    def files(self, *path): return (file for file in os.listdir(self.repository, *path) if os.path.isfile(file))
    def folder(self, *path): return os.path.join(self.repository, *path)
    def file(self, *path): return os.path.join(self.repository, *path)

    def read(self, *args, file, filetype, **kwargs):
        with self.mutex[str(file)]:
            content = load(*args, file=file, filetype=filetype, **kwargs)
            return content

    def write(self, content, *args, file, filemode, **kwargs):
        with self.mutex[str(file)]:
            save(content, *args, file=file, mode=filemode, **kwargs)

    @property
    def repository(self): return self.__repository
    @property
    def capacity(self): return self.__capacity
    @property
    def mutex(self): return self.__mutex


class DataframeFile(File):
    def __init__(self, *args, datetypes=[], datatypes={}, **kwargs):
        assert isinstance(datetypes, list) and isinstance(datatypes, dict)
        super().__init__(*args, **kwargs)
        self.__datetypes = datetypes
        self.__datatypes = datatypes

    def read(self, *args, **kwargs):
        parameters = dict(datetypes=self.datetypes, datatypes=self.datatypes, filetype=pd.DataFrame)
        return super().read(*args, **parameters, **kwargs)

    def write(self, *args, **kwargs):
        parameters = dict(datetypes=self.datetypes, datatypes=self.datatypes, filetype=pd.DataFrame)
        super().write(*args, **parameters, **kwargs)

    @property
    def datetypes(self): return self.__datetypes
    @property
    def datatypes(self): return self.__datatypes


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
        filetype = type(content)
        fileext = str(os.path.splitext(file)[-1]).strip(".")
        try:
            function = retrieve(filetype, fileext)
            return function(content, *args, file=file, **kwargs)
        except IndexError:
            return mainfunction(content, *args, file=file, **kwargs)

    def load_wrapper(*args, filetype, file, **kwargs):
        fileext = str(os.path.splitext(file)[-1]).strip(".")
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
    filetype = type(content)
    fileext = str(os.path.splitext(file)[-1]).strip(".")
    raise ValueError(filetype, fileext)

@save.register(xr.Dataset, "nc")
def save_netcdf(content, *args, file, mode, **kwargs):
    xr.Dataset.to_netcdf(content, file, mode=mode, compute=True)

@save.register(pd.DataFrame, "csv")
@save.register(dk.DataFrame, "csv")
def save_csv(content, *args, file, mode, **kwargs):
    header = not os.path.isfile(file) or mode == "w"
    parms = dict(index=False, header=header)
    if isinstance(content, dk.DataFrame):
        update = dict(compute=True, single_file=True, header_first_partition_only=True)
        parms.update(update)
    content.to_csv(file, mode=mode, **parms)

@save.register(pd.DataFrame, "hdf")
@save.register(dk.DataFrame, "hdf")
def save_hdf5(self, content, *args, file, group=None, mode, **kwargs):
    parms = dict(format="fixed", append=False)
    content.to_hdf(file, group, mode=mode, **parms)


@dispatcher
def load(*args, filetype, file, **kwargs):
    fileext = str(os.path.splitext(file)[-1]).strip(".")
    raise ValueError(filetype, fileext)

@load.register(xr.Dataset, "nc")
def load_netcdf(*args, file, partitions=None, **kwargs):
    return xr.open_dataset(file, chunks=partitions)

@load.register(dk.DataFrame, "csv")
def load_csv_delayed(*args, file, size, datatypes={}, datetypes=[], **kwargs):
    parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
    return dk.read_csv(file, blocksize=size, **parms)

@load.register(pd.DataFrame, "csv")
def load_csv_immediate(*args, file, datatypes={}, datetypes=[], **kwargs):
    parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
    return pd.read_csv(file,  iterator=False, **parms)

@load.register(pd.DataFrame, "hdf")
def load_hdf5(*args, file, group=None, **kwargs):
    return pd.read_csv(file, key=group, iterator=False)










