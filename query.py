# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   Query Object
@author: Jack Kirby Cook

"""

import inspect

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Header", "Query"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class Header(object):
    def __init__(self, *args, **kwargs): pass


class QueryMeta(type):
    def __call__(cls, *args, **kwargs):
        def decorator(generator):
            assert inspect.isgeneratorfunction(generator)
            instance = super(QueryMeta, cls).__call__(generator, *args, **kwargs)
            return instance
        return decorator


class Query(object, metaclass=QueryMeta):
    def __init__(self, generator, *arguments, **parameters):
        pass

    def __call__(self, contents, *args, **kwargs):
        pass

    @property
    def generator(self): return self.__generator
    @property
    def outlet(self): return self.__outlet
    @property
    def inlet(self): return self.__inlet




