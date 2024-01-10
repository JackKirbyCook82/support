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

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Routine"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class Routine(threading.Thread):
    def __repr__(self): return self.name
    def __bool__(self): return super().is_alive()

    def __init__(self, routine, *args, **kwargs):
        assert callable(routine)
        name = kwargs.get("name", self.__class__.__name__)
        threading.Thread.__init__(self, name=name, daemon=False)
        self.__routine = routine
        self.__arguments = list()
        self.__parameters = dict()
        self.__results = None

    def setup(self, *args, **kwargs):
        self.arguments.extend(list(args)) if args else False
        self.parameters.update(dict(kwargs)) if kwargs else False
        return self

    def start(self, *args, **kwargs):
        LOGGER.info(f"Started: {repr(self)}")
        threading.Thread.start(self)
        return self

    def join(self, *args, **kwargs):
        threading.Thread.join(self)
        LOGGER.info(f"Stopped: {repr(self)}")
        return self

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

    def process(self, *args, **kwargs):
        routine = self.routine.__call__ if hasattr(self.routine, "__call__") else self.routine
        results = routine(*args, **kwargs)
        generator = results if isinstance(results, types.GeneratorType) else iter([results])
        self.results = list(generator)

    @property
    def routine(self): return self.__routine
    @property
    def arguments(self): return self.__arguments
    @property
    def parameters(self): return self.__parameters
    @property
    def results(self): return self.__results
    @results.setter
    def results(self, results): self.__results = results





