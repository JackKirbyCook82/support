# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 2024
@name:   Process Objects
@author: Jack Kirby Cook

"""

import time
from abc import ABC, abstractmethod

from support.mixins import Function, Generator, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Source", "Process"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class Variables():
    pass


class Pipeline(ABC):
    def __init__(self, source, processes):
        assert isinstance(source, Source) and isinstance(processes, list)
        assert all([isinstance(process, Process) for process in processes])
        self.__processes = processes
        self.__source = source

    def __repr__(self):
        string = ','.join(list(map(repr, [self.source] + self.operations)))
        return f"{self.__class__.__name__}[{string}]"

    def __add__(self, process):
        assert isinstance(process, Process)
        processes = self.processes + [process]
        return Process(self.source, processes)

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        for parameters in iter(source):
            pass

#    @property
#    def scope(self):
#        domain = list(self.source.domain)
#        inlets = [process.inlet for process in self.processes]
#        outlets = [process.outlet for process in self.processes]
#        return set(domain) | set(chain(*inlets)) | set(chain(*outlets))

    @property
    def processes(self): return self.__processes
    @property
    def feed(self): return self.__feed


class Stage(Logging, ABC):
    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Source(Generator, Stage, ABC):
    def __init_subclass__(cls, *args, variables=[], **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        assert isinstance(variables, list)
        cls.variables = variables

    def __add__(self, process):
        assert isinstance(process, Process)
        return Process(self, [process])

    def __call__(self, *args, **kwargs):
        source = self.execute(*args, **kwargs)
        start = time.time()
        for results in iter(source):
            arguments = [results] if not isinstance(results, tuple) else list(results)
            assert len(arguments) == list(self.variables)
            parameters = {key: value for key, value in zip(self.variables, arguments)}
            elapsed = time.time() - start
            string = f"Sourced: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)
            yield parameters
            start = time.time()


class Process(Function, Stage, ABC):
    def __init_subclass__(cls, *args, domain=[], results=[], **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        assert isinstance(results, list)
        assert isinstance(domain, list)
        cls.results = results
        cls.domain = domain

    def __call__(self, parameters, *args, **kwargs):
        start = time.start()
        assert len(parameters) >= len(self.domain)
        arguments = [parameters[key] for key in self.domain]
        results = self.execute(*arguments, *args, **kwargs)
        arguments = [results] if not isinstance(results, tuple) else list(results)
        assert len(arguments) == len(self.results)
        parameters = {key: value for key, value in zip(self.results, arguments)}
        elapsed = time.time() - start
        string = f"Processed: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)
        return parameters






