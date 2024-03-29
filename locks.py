# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   Lock Objects
@author: Jack Kirby Cook

"""

import multiprocessing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Lock", "Locks"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class Lock(object):
    def __bool__(self): return not self.mutex.locked()
    def __repr__(self): return self.name

    def __init__(self, *args, key=None, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__mutex = multiprocessing.RLock()
        self.__timeout = timeout
        self.__key = key

    def __enter__(self):
        self.mutex.acquire(timeout=self.timeout)
        return self

    def __exit__(self, error_type, error_value, error_traceback):
        self.mutex.release()

    @property
    def timeout(self): return self.__timeout
    @property
    def mutex(self): return self.__mutex
    @property
    def name(self): return self.__name
    @property
    def key(self): return self.__key


class Locks(dict):
    def __repr__(self): return self.name
    def __init__(self, *args, timeout=None, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__timeout = timeout

    def __getitem__(self, key):
        if key not in self.keys():
            name = str(self.name).rstrip("s")
            timeout = -1 if self.timeout is None else self.timeout
            self[key] = Lock(key=key, name=name, timeout=timeout)
        return super().__getitem__(key)

    @property
    def timeout(self): return self.__timeout
    @property
    def name(self): return self.__name



