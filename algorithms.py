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


class ProcessError(Exception):
    def __str__(self): return f"{type(self).__name__}|{repr(self.algorithm)}"
    def __init__(self, algorithm): self.algorithm = algorithm

class PriorProcessError(ProcessError): pass
class PostProcessError(ProcessError): pass


class Stage(Logging, ABC):
    @abstractmethod
    def execute(self, *args, **kwargs): pass


class Source(Stage, ABC):
    def __init_subclass__(cls, *args, signature, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        assert isinstance(signature, str)
        outlet = str(signature).translate("()->")
        cls.range = str(outlet).split(",")

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
    def __init_subclass__(cls, *args, signature, **kwargs):
        try: super().__init_subclass__(*args, **kwargs)
        except TypeError: super().__init_subclass__()
        assert isinstance(signature, str)
        inlet, outlet = str(signature).translate("()").split("->")
        cls.domain = str(inlet).split(",")
        cls.range = str(outlet).split(",")

    def __call__(self, parameters, *args, **kwargs):
        assert isinstance(parameters, dict)
        if not all([key in parameters for key in self.domain]): PriorProcessError(self)
        domain = list(map(parameters.get, self.domain))
        start = time.time()
        contents = self.execute(*domain, *args, **kwargs)
        if contents is None: raise PostProcessError(self)
        contents = [contents] if not isinstance(contents, tuple) else list(contents)
        assert len(contents) == len(self.range)
        elapsed = time.time() - start
        string = f"Calculated: {repr(self)}[{elapsed:.02f}s]"
        self.logger.info(string)
        parameters = dict(zip(self.range, contents))
        return parameters














