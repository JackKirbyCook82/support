# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 2024
@name:   Process Objects
@author: Jack Kirby Cook

"""

import time
import logging
from itertools import chain
from abc import ABC, abstractmethod

from support.mixins import Function, Generator, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Source", "Process"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Variables(object):
    def __iter__(self):
        for variable, value in self.domain.items():
            if value is None: continue
            else: yield variable, value

    def __init__(self, domain):
        self.domain = dict.fromkeys(domain)

    def __call__(self, domain):
        variables = set(self.domain) - domain
        for variable in variables:
            del self.domain[variable]
        return self

    def __iadd__(self, parameters):
        assert isinstance(parameters, dict)
        variables = set(self.domain)
        for variable in variables:
            value = parameters.get(variable, None)
            self.domain[variable] = value
        return self


class ProcessErrorMeta(type):
    def __init__(cls, name, bases, attrs, *args, title=None, **kwargs):
        assert str(name).endswith("Error")
        super(ProcessErrorMeta, cls).__init__(name, bases, attrs)
        cls.__logger__ = __logger__
        cls.__title__ = title

    def __call__(cls, process):
        logger, title, name = cls.__logger__, cls.__title__, cls.__name__
        instance = super(ProcessErrorMeta, cls).__call__(name, process)
        logger.info(f"{cls.title}: {repr(instance.process)}")
        return instance


class ProcessError(Exception, metaclass=ProcessErrorMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __str__(self): return f"{self.name}|{repr(self.page)}"
    def __init__(self, name, process):
        self.__process = process
        self.__name = name

    @property
    def process(self): return self.__process
    @property
    def name(self): return self.__name


class CeasedProcessError(ProcessError, title="Ceased"): pass
class TerminatedProcessError(ProcessError, title="Terminated"): pass


class Pipeline(ABC):
    def __init__(self, source, processes):
        assert isinstance(source, Source) and isinstance(processes, list)
        assert all([isinstance(process, Process) for process in processes])
        self.__processes = processes
        self.__source = source

    def __repr__(self):
        string = ', '.join(list(map(repr, [self.source] + self.processes)))
        return f"{self.__class__.__name__}[{string}]"

    def __add__(self, process):
        assert isinstance(process, Process)
        processes = self.processes + [process]
        return Pipeline(self.source, processes)

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        for parameters in source:
            domain = self.domain(index=0)
            variables = Variables(domain)
            variables += parameters
            for index, process in enumerate(self.processes):
                domain = self.domain(index=index)
                variables = variables(domain)
                parameters = dict(variables)
                try: parameters = process(parameters, *args, **kwargs)
                except CeasedProcessError: break
                except TerminatedProcessError: break
                variables += parameters
            del variables

    def domain(self, index=0):
        domain = [process.domain for process in self.processes[index:]]
        domain = set(chain(*domain))
        return domain

    @property
    def processes(self): return self.__processes
    @property
    def source(self): return self.__source


class Stage(Logging, ABC):
    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Source(Generator, Stage, ABC):
    def __init_subclass__(cls, *args, arguments=[], **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        assert isinstance(arguments, list)
        cls.arguments = arguments

    def __add__(self, process):
        assert isinstance(process, Process)
        return Pipeline(self, [process])

    def __call__(self, *args, **kwargs):
        source = self.execute(*args, **kwargs)
        start = time.time()
        for results in iter(source):
            assert results is not None
            arguments = [results] if not isinstance(results, tuple) else list(results)
            assert len(arguments) == len(self.arguments)
            elapsed = time.time() - start
            string = f"Sourced: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)
            parameters = dict(zip(self.arguments, arguments))
            yield parameters
            start = time.time()


class Process(Function, Stage, ABC):
    def __init_subclass__(cls, *args, domain=[], arguments=[], **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        assert isinstance(arguments, list)
        assert isinstance(domain, list)
        cls.arguments = arguments
        cls.domain = domain

    def __call__(self, parameters, *args, **kwargs):
        assert isinstance(parameters, dict)
        if not all([key in parameters for key in self.domain]): CeasedProcessError(self)
        domain = list(map(parameters.get, self.domain))
        start = time.time()
        results = self.execute(*domain, *args, **kwargs)
        if results is None: raise TerminatedProcessError(self)
        arguments = [results] if not isinstance(results, tuple) else list(results)
        assert len(arguments) == len(self.arguments)
        elapsed = time.time() - start
        string = f"Processed: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)
        parameters = dict(zip(self.arguments, arguments))
        return parameters





