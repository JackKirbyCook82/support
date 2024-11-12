# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 2024
@name:   Process Objects
@author: Jack Kirby Cook

"""

import time
import types
from itertools import chain
from abc import ABC, abstractmethod

from support.mixins import Function, Generator, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Source", "Process"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class Variables(object):
    def __init__(self, contents): self.contents = contents
    def __len__(self): return len(self.contents)

    def __contains__(self, keys):
        assert isinstance(keys, list) and len(keys) <= len(self)
        return all([key in self.contents for key in keys])

    def __getitem__(self, keys): return self.get(keys)
    def __setitem__(self, keys, values):
        assert isinstance(keys, list) and isinstance(values, list) and len(keys) == len(values) and len(keys) <= len(self) and list(keys) in self
        contents = dict(zip(keys, values))
        self.contents.update(contents)

    def get(self, keys):
        assert isinstance(keys, list) and len(keys) <= len(self) and keys in self
        values = list(map(self.contents.get, keys))
        contents = dict(zip(keys, values))
        return contents

    def put(self, contents):
        assert isinstance(contents, dict) and len(contents) <= len(self) and list(contents) in self
        self.contents.update(contents)

    @classmethod
    def define(cls, domain): return cls(dict.fromkeys(domain))
    def redefine(self, domain):
        assert len(domain) <= len(self) and list(domain) in self
        for key in set(self.contents) - set(domain):
            del self.contents[key]


class Pipeline(ABC):
    def __init__(self, source, processes):
        assert isinstance(source, Source) and isinstance(processes, list)
        assert all([isinstance(process, Process) for process in processes])
        self.__processes = processes
        self.__source = source

    def __repr__(self):
        string = ','.join(list(map(repr, [self.source] + self.processes)))
        return f"{self.__class__.__name__}[{string}]"

    def __add__(self, process):
        assert isinstance(process, Process)
        processes = self.processes + [process]
        return Pipeline(self.source, processes)

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        for parameters in source:
            assert isinstance(parameters, dict)
            domain = self.domain(index=0)
            variables = Variables.define(domain)
            variables.put(parameters)
            for index, process in enumerate(self.processes):
                domain = self.domain(index=index)
                variables.redefine(domain)
                parameters = variables.get(process.domain)
                parameters = process(parameters, *args, **kwargs)
                variables.put(parameters)
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

            if isinstance(results, types.NoneType):
                pass

            arguments = [results] if not isinstance(results, tuple) else list(results)
            assert len(arguments) == len(self.arguments)
            parameters = {key: value for key, value in zip(self.arguments, arguments)}
            elapsed = time.time() - start
            string = f"Sourced: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)
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
        start = time.time()
        assert isinstance(parameters, dict)
        assert len(parameters) >= len(self.domain)
        domain = [parameters[key] for key in self.domain]
        results = self.execute(*domain, *args, **kwargs)

        if isinstance(results, types.NoneType):
            pass

        arguments = [results] if not isinstance(results, tuple) else list(results)
        assert len(arguments) == len(self.arguments)
        parameters = {key: value for key, value in zip(self.arguments, arguments)}
        elapsed = time.time() - start
        string = f"Processed: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)
        return parameters






