# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Synchronization Objects
@author: Jack Kirby Cook

"""

import sys
import time
import inspect
import traceback
import threading

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["RoutineThread", "RepeatingThread"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Thread(Logging):
    def __bool__(self): return bool(self.active)
    def __init__(self, routine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(routine)
        self.__routine = routine
        self.__arguments = list()
        self.__parameters = dict()
        self.__active = False

    def setup(self, *arguments, **parameters):
        self.arguments.extend(list(arguments)) if arguments else False
        self.parameters.update(dict(parameters)) if parameters else False
        return self

    def run(self):
        self.active = True
        try:
            self.console(title="Running")
            self.process(*self.arguments, **self.parameters)
        except BaseException as error:
            string = str(error.__class__.__name__)
            self.console(string, title="Error")
            error_type, error_value, error_traceback = sys.exc_info()
            traceback.print_exception(error_type, error_value, error_traceback)
        else:
            self.console(title="Completed")
        self.active = False

    def process(self, *args, **kwargs):
        routine = self.routine.__call__ if hasattr(self.routine, "__call__") else self.routine
        if not inspect.isgeneratorfunction(routine): routine(*args, **kwargs)
        else: list(routine(*args, **kwargs))

    @property
    def active(self): return self.__active
    @active.setter
    def active(self, active): self.__active = active
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
        self.console(title="Started")
        threading.Thread.start(self)

    def join(self, *args, **kwargs):
        threading.Thread.join(self)
        self.console(title="Stopped")


class RepeatingThread(Thread, threading.Thread):
    def __init__(self, *args, wait=None, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        threading.Thread.__init__(self, name=self.name, daemon=False)
        self.__mutex = threading.RLock()
        self.__repeating = True
        self.__wait = wait

    def process(self, *args, **kwargs):
        while bool(self.repeating):
            super().process(*args, **kwargs)
            if self.wait is not None:
                time.sleep(self.wait)

    def cease(self, *args, **kwargs):
        with self.mutex:
            self.console(title="Ceased")
            self.repeating = False

    @property
    def repeating(self): return self.__repeating
    @repeating.setter
    def repeating(self, repeating): self.__repeating = repeating

    @property
    def mutex(self): return self.__mutex
    @property
    def wait(self): return self.__wait



