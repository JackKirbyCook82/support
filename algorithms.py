# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 2024
@name:   Algorithm Objects
@author: Jack Kirby Cook

"""

import time
from abc import ABC, abstractmethod

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Algorithm", "Source"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class AlgorithmError(Exception):
    def __str__(self): return f"{type(self).__name__}|{repr(self.algorithm)}"
    def __init__(self, algorithm): self.algorithm = algorithm

class PriorAlgorithmError(AlgorithmError): pass
class PostAlgorithmError(AlgorithmError): pass


class Pipeline(object):
    def __init__(self, source, algorithms):
        assert isinstance(source, Source) and isinstance(algorithms, list)
        assert all([isinstance(algorithm, Algorithm) for algorithm in algorithms])
        self.__algorithms = algorithms
        self.__source = source

    def __repr__(self):
        string = ', '.join(list(map(repr, [self.source] + self.algorithms)))
        return f"{type(self).__name__}[{string}]"

    def __add__(self, other):
        assert isinstance(other, Algorithm)
        algorithms = self.algorithms + [other]
        return Pipeline(self.source, algorithms)

    def __call__(self, *args, **kwargs):
        source = self.source(*args, **kwargs)
        for parameters in iter(source):
            for algorithm in self.algorithms:
                try: update = algorithm(parameters, *args, **kwargs)
                except PriorAlgorithmError: break
                except PostAlgorithmError: break
                else: parameters.update(update)
                finally: pass
            del parameters

    @property
    def algorithms(self): return self.__algorithms
    @property
    def source(self): return self.__source


class Stage(Logging, ABC):
    def __init_subclass__(cls, *args, signature=None, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        if not bool(signature): return
        assert isinstance(signature, str)
        inlet, outlet = str(signature).replace("", "").split("->")
        inlet, outlet = [str(value).strip("()").split(",") for value in (inlet, outlet)]
        cls.domain = list(filter(bool, inlet))
        cls.range = list(filter(bool, outlet))

    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Source(Stage, ABC):
    def __add__(self, other):
        assert isinstance(other, Algorithm)
        return Pipeline(self, [other])

    def __call__(self, *args, **kwargs):
        source = self.execute(*args, **kwargs)
        start = time.time()
        for contents in iter(source):
            assert contents is not None
            contents = [contents] if not isinstance(contents, tuple) else list(contents)
            assert len(contents) == len(self.range)
            elapsed = time.time() - start
            string = f"Sourced: {repr(self)}[{elapsed:.02f}s]"
            self.logger.info(string)
            parameters = dict(zip(self.range, contents))
            yield parameters
            start = time.time()


class Algorithm(Stage, ABC):
    def __call__(self, parameters, *args, **kwargs):
        assert isinstance(parameters, dict)
        if not all([key in parameters for key in self.domain]): PriorAlgorithmError(self)
        domain = list(map(parameters.get, self.domain))
        start = time.time()
        contents = self.execute(*domain, *args, **kwargs)
        if contents is None: raise PostAlgorithmError(self)
        contents = [contents] if not isinstance(contents, tuple) else list(contents)
        assert len(contents) == len(self.range)
        elapsed = time.time() - start
        string = f"Computed: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)
        parameters = dict(zip(self.range, contents))
        return parameters














