# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Functions/Objects
@author: Jack Kirby Cook

"""

import os
import logging
import multiprocessing
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from abc import ABC, ABCMeta
from functools import update_wrapper
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["File", "Archive"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class File(ntuple("File", "filename fileheader filetype")):
    def __str__(self): return os.path.splitext(self.filename)[0]

    def load(self, *args, folder, mode, **kwargs):
        file = os.path.join(folder, self.filename)
        return load(*args, file=file, mode=mode, **self.parameters, **kwargs) if os.path.exists(file) else None

    def save(self, content, *args, folder, mode, **kwargs):
        file = os.path.join(folder, self.filename)
        save(content, *args, file=file, mode=mode, **self.parameters, **kwargs)
        __logger__.info("Saved: {}".format(str(file)))

    @property
    def parameters(self): return dict(types=self.types, dates=self.dates, columns=self.columns, type=self.filetype)
    @property
    def types(self): return {key: value for key, value in self.fileheader.items() if not any([value is str, value is np.datetime64])}
    @property
    def dates(self): return [key for key, value in self.fileheader.items() if value is np.datetime64]
    @property
    def columns(self): return list(self.fileheader.keys())
    @property
    def extension(self): return os.path.splitext(self.filename)[1]
    @property
    def name(self): return os.path.splitext(self.filename)[0]


class ArchiveMeta(ABCMeta):
    locks = {}

    def __init__(cls, *args, **kwargs):
        existing = {name: file for name, file in getattr(cls, "__files__", {}).items()}
        update = {str(file): file for file in kwargs.get("files", {})}
        cls.__files__ = existing | update

    def __call__(cls, *args, repository, **kwargs):
        lock = ArchiveMeta.locks.get(str(repository),  multiprocessing.RLock())
        ArchiveMeta.locks[str(repository)] = lock
        instance = super(ArchiveMeta, cls).__call__(*args, repository=repository, files=cls.__files__, lock=lock, **kwargs)
        return instance


class Archive(ABC, metaclass=ArchiveMeta):
    def __init_subclass__(cls, *args, **kwargs): pass

    def __repr__(self): return self.__class__.__name__
    def __bool__(self): return not self.empty if self.table is not None else False
    def __init__(self, *args, repository, files, lock, **kwargs):
        self.__repository = repository
        self.__files = files
        self.__mutex = lock

    def __eq__(self, other): return self.repository == other.repository
    def __ne__(self, other): return not self.__eq__(other)
    def __add__(self, other):
        assert isinstance(other, Archive) and self == other
        name = repr(self).strip("Archive") + repr(other).strip("Archive") + "Archive"
        files = list((self.files | other.files).values())
        ArchiveType = type(name, (Archive,), {}, files=files)
        return ArchiveType(repository=self.repository)

    def load(self, *args, folder, mode="r", **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        if not os.path.exists(folder):
            return {}
        with self.mutex:
            contents = {name: file.load(*args, folder=folder, mode=mode, **kwargs) for name, file in self.files.items()}
            contents = {name: content for name, content in contents.items() if content is not None}
            return contents

    def save(self, contents, *args, folder, mode, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        if not os.path.exists(folder):
            os.makedirs(folder)
        with self.mutex:
            contents = {name: content for name, content in contents.items() if content is not None}
            contents = {name: content for name, content in contents.items() if name in self.files.keys()}
            for name, content in contents.items():
                self.files[name].save(content, *args, folder=folder, mode=mode, **kwargs)

    @property
    def directory(self): return os.listdir(self.repository)
    @property
    def empty(self): return not bool(os.listdir(self.repository))
    @property
    def size(self): return len(os.listdir(self.repository))

    @property
    def repository(self): return self.__repository
    @property
    def files(self): return self.__files
    @property
    def mutex(self): return self.__mutex


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









