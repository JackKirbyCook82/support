# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Synchronization Objects
@author: Jack Kirby Cook

"""

import sys
import types
import logging
import traceback
import threading
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["MainThread", "SideThread"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class Interface(ABC):
    def __repr__(self): return self.name
    def __init__(self, routine, *args, **kwargs):
        assert callable(routine)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__routine = routine
        self.__arguments = list()
        self.__parameters = dict()
        self.__results = None

    def setup(self, *args, **kwargs):
        self.arguments.extend(list(args)) if args else False
        self.parameters.update(dict(kwargs)) if kwargs else False

    def run(self):
        try:
            LOGGER.info(f"Running: {repr(self)}")
            self.process(*self.arguments, **self.parameters)
        except BaseException as error:
            LOGGER.error(f"Error: {repr(self)}[{error.__class__.__name__}]")
            error_type, error_value, error_traceback = sys.exc_info()
            traceback.print_exception(error_type, error_value, error_traceback)
        else:
            LOGGER.info(f"Completed: {repr(self)}")

    @abstractmethod
    def process(self, *args, **kwargs): pass

    @property
    def results(self): return self.__results
    @results.setter
    def results(self, results): self.__results = results
    @property
    def arguments(self): return self.__arguments
    @property
    def parameters(self): return self.__parameters
    @property
    def routine(self): return self.__routine
    @property
    def name(self): return self.__name


class MainThread(Interface):
    def process(self, *args, **kwargs):
        routine = self.routine.__call__ if hasattr(self.routine, "__call__") else self.routine
        self.results = routine(*args, **kwargs)


class SideThread(Interface, threading.Thread):
    def __init__(self, *args, **kwargs):
        name = kwargs.get("name", self.__class__.__name__)
        threading.Thread.__init__(self, name=name, daemon=False)
        Interface.__init__(self, *args, **kwargs)

    def start(self, *args, **kwargs):
        LOGGER.info(f"Started: {repr(self)}")
        threading.Thread.start(self)

    def join(self, *args, **kwargs):
        threading.Thread.join(self)
        LOGGER.info(f"Stopped: {repr(self)}")

    def process(self, *args, **kwargs):
        routine = self.routine.__call__ if hasattr(self.routine, "__call__") else self.routine
        results = routine(*args, **kwargs)
        generator = results if isinstance(results, types.GeneratorType) else iter([results])
        self.results = list(generator)





