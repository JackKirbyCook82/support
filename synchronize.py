# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Synchronization Objects
@author: Jack Kirby Cook

"""

import sys
import time
import types
import logging
import traceback
import threading

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Breaker", "MainThread", "SideThread", "CycleThread"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Breaker(object):
    pass


class Thread(object):
    def __repr__(self): return self.name
    def __bool__(self): return bool(self.active)
    def __init__(self, routine, *args, **kwargs):
        assert callable(routine)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__active = threading.Event()
        self.__routine = routine
        self.__arguments = list()
        self.__parameters = dict()
        self.__results = []

    def setup(self, *args, **kwargs):
        self.arguments.extend(list(args)) if args else False
        self.parameters.update(dict(kwargs)) if kwargs else False

    def run(self):
        try:
            __logger__.info(f"Running: {repr(self)}")
            self.process(*self.arguments, **self.parameters)
        except BaseException as error:
            __logger__.error(f"Error: {repr(self)}[{error.__class__.__name__}]")
            error_type, error_value, error_traceback = sys.exc_info()
            traceback.print_exception(error_type, error_value, error_traceback)
        else:
            __logger__.info(f"Completed: {repr(self)}")
        finally:
            self.active.clear()

    def process(self, *args, **kwargs):
        routine = self.routine.__call__ if hasattr(self.routine, "__call__") else self.routine
        results = routine(*args, **kwargs)
        generator = results if isinstance(results, types.GeneratorType) else iter([results])
        results = list(filter(None, generator))
        assert isinstance(results, list)
        self.results.append(results)

    @property
    def arguments(self): return self.__arguments
    @property
    def parameters(self): return self.__parameters
    @property
    def routine(self): return self.__routine
    @property
    def results(self): return self.__results
    @property
    def active(self): return self.__active
    @property
    def name(self): return self.__name


class MainThread(Thread):
    pass


class SideThread(Thread, threading.Thread):
    def __init__(self, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        threading.Thread.__init__(self, name=self.name, daemon=False)

    def start(self, *args, **kwargs):
        __logger__.info(f"Started: {repr(self)}")
        threading.Thread.start(self)

    def cease(self, *args, **kwargs): pass
    def join(self, *args, **kwargs):
        threading.Thread.join(self)
        __logger__.info(f"Stopped: {repr(self)}")


class CycleThread(SideThread):
    def __init__(self, *args, wait=None, **kwargs):
        SideThread.__init__(self, *args, **kwargs)
        self.__cycling = threading.Event()
        self.__wait = wait

    def process(self, *args, **kwargs):
        while bool(self.cycling):
            super().process(*args, **kwargs)
            if self.wait is not None:
                time.sleep(self.wait)

    def cease(self, *args, **kwargs):
        __logger__.info(f"Ceased: {repr(self)}")
        self.cycling.clear()

    @property
    def cycling(self): return self.__cycling
    @property
    def wait(self): return self.__wait



