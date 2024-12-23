# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Synchronization Objects
@author: Jack Kirby Cook

"""

import sys
import time
import inspect
import logging
import traceback
import threading

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["RoutineThread", "RepeatingThread"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Thread(Logging):
    def __bool__(self): return bool(self.active)
    def __init__(self, routine, *args, **kwargs):
        assert callable(routine)
        Logging.__init__(self, *args, **kwargs)
        self.__logger = __logger__
        self.__routine = routine
        self.__arguments = list()
        self.__parameters = dict()
        self.__active = False
        self.__results = None

    def setup(self, *arguments, **parameters):
        assert self.results is None
        self.arguments.extend(list(arguments)) if arguments else False
        self.parameters.update(dict(parameters)) if parameters else False
        return self

    def run(self):
        self.active = True
        try:
            self.logger.info(f"Running: {repr(self)}")
            self.process(*self.arguments, **self.parameters)
        except BaseException as error:
            self.logger.error(f"Error: {repr(self)}[{error.__class__.__name__}]")
            error_type, error_value, error_traceback = sys.exc_info()
            traceback.print_exception(error_type, error_value, error_traceback)
        else:
            self.logger.info(f"Completed: {repr(self)}")
        self.active = False

    def process(self, *args, **kwargs):
        routine = self.routine.__call__ if hasattr(self.routine, "__call__") else self.routine
        if not inspect.isgeneratorfunction(routine):
            self.results = routine(*args, **kwargs)
        else:
            generator = routine(*args, **kwargs)
            self.results = list(generator)

    @property
    def active(self): return self.__active
    @active.setter
    def active(self, active): self.__active = active
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


class RoutineThread(Thread, threading.Thread):
    def __init__(self, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        threading.Thread.__init__(self, name=self.name, daemon=False)

    def start(self, *args, **kwargs):
        self.logger.info(f"Started: {repr(self)}")
        threading.Thread.start(self)

    def cease(self, *args, **kwargs): pass
    def join(self, *args, **kwargs):
        threading.Thread.join(self)
        self.logger.info(f"Stopped: {repr(self)}")


class RepeatingThread(Thread, threading.Thread):
    def __init__(self, *args, wait=None, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        threading.Thread.__init__(self, name=self.name, daemon=False)
        self.__cycling = True
        self.__wait = wait

    def process(self, *args, **kwargs):
        while bool(self.cycling):
            super().process(*args, **kwargs)
            if self.wait is not None:
                time.sleep(self.wait)

    def cease(self, *args, **kwargs):
        self.logger.info(f"Ceased: {repr(self)}")
        self.cycling = False

    @property
    def cycling(self): return self.__cycling
    @cycling.setter
    def cycling(self, cycling): self.__cycling = cycling
    @property
    def wait(self): return self.__wait



