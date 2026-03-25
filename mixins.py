# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Mixins Object
@author: Jack Kirby Cook

"""

import logging
from abc import ABC

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Mixin", "Logging"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Mixin(ABC):
    def __init_subclass__(cls, **kwargs):
        try: super().__init_subclass__(**kwargs)
        except TypeError: super().__init_subclass__()

    def __new__(cls, *args, **kwargs):
        try: return super().__new__(cls, *args, **kwargs)
        except TypeError: return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()


class Logging(Mixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__logger = __logger__

    def console(self, event, *strings):
        string = f"{str(event)}: {str(self.name)}"
        if bool(strings):
            strings = ", ".join(list(strings))
            string = ": ".join([string, strings])
        self.logger.info(string)

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name












