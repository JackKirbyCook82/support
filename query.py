# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   Query Object
@author: Jack Kirby Cook

"""

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Header", "Input", "Output", "Query"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class Header(object): pass
class Input(object): pass
class Output(object): pass


class QueryMeta(type): pass
class Query(object, metaclass=QueryMeta): pass




