# -*- coding: utf-8 -*-
"""
Created on Tues Nov 17 2026
@name    Pipeline Objects
@author: Jack Kirby Cook
@file:   support/pipeline.py

"""

from abc import ABC
from types import NoneType

from support.decorators import Decorator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Pipelines", "Errors"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class PipelineError(Exception): pass
class InletError(PipelineError): pass
class OutletError(PipelineError): pass
class MissingInletError(InletError): pass
class EmptyInletError(InletError): pass
class MissingOutletError(OutletError): pass
class EmptyOutletError(OutletError): pass


class Pipeline(Decorator, ABC):
    def __init__(self, *args, locator, **kwargs):
        assert isinstance(locator, (int, str))
        super().__init__(*args, **kwargs)
        self.__locator = locator

    def locate(self, *args, **kwargs):
        if isinstance(self.locator, str): inlet = kwargs[self.locator]
        elif isinstance(self.locator, int): inlet = args[self.locator]
        else: raise TypeError(type(self.locator))
        return inlet

    @property
    def locator(self): return self.__locator


class PipelineFunction(Pipeline):
    def decorator(self, instance, *args, **kwargs):
        inlet = self.locate(*args, **kwargs)
        if isinstance(inlet, NoneType): raise MissingInletError()
        elif len(inlet) == 0: raise EmptyInletError()
        outlet = self.invoke(instance, *args, **kwargs)
        if isinstance(outlet, NoneType): raise MissingOutletError()
        elif len(outlet) == 0: raise EmptyOutletError()
        return outlet


class PipelineGenerator(Pipeline):
    def decorator(self, instance, *args, **kwargs):
        inlet = self.locate(*args, **kwargs)
        if isinstance(inlet, NoneType): raise MissingInletError()
        elif len(inlet) == 0: raise EmptyInletError()
        generator = self.invoke(instance, *args, **kwargs)
        for outlet in generator:
            if isinstance(outlet, NoneType): raise MissingOutletError()
            elif len(outlet) == 0: raise EmptyOutletError()
            yield outlet


class Pipelines:
    Function = PipelineFunction
    Generator = PipelineGenerator

class Errors:
    class Inlet:
        Missing = MissingInletError
        Empty = EmptyInletError
    class Outlet:
        Missing = MissingOutletError
        Empty = EmptyOutletError

