# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Mixins Object
@author: Jack Kirby Cook
@file:   support/mixins.py

"""

import logging
from abc import ABC

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Mixin", "LoggingMixin"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class MixinError(Exception): pass
class Mixin(ABC):
    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        assert isinstance(mro, tuple)
        index = mro.index(Mixin)
        remaining = mro[index + 1:]
        if remaining != (ABC, object): raise MixinError()

    def __new__(cls, *args, **kwargs): return super().__new__(cls)
    def __init__(self, *args, **kwargs): super().__init__()


class LoggingMixin(Mixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = self.__class__.__name__
        self.__name = kwargs.get("name", name)
        self.__logger = __logger__

    def console(self, *strings, title, **kwargs):
        string = ", ".join(map(str, strings))
        string = f"{str(title)}: {str(self.name)}: {str(string)}"
        self.logger.info(string)

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name

