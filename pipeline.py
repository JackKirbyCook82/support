# -*- coding: utf-8 -*-
"""
Created on Tues Nov 17 2026
@name    Pipeline Objects
@author: Jack Kirby Cook
@file:   support/pipeline.py

"""

from abc import ABC

from support.decorators import Decorator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Pipelines"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Pipeline(Decorator, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PipelineFunction(Pipeline):
    def decorator(self, instance, inlet, *args, **kwargs):
        outlet = self.function(instance, inlet, *args, **kwargs)
        return outlet


class PipelineGenerator(Pipeline):
    def decorator(self, instance, inlet, *args, **kwargs):
        generator = self.function(instance, inlet, *args, **kwargs)
        for outlet in generator:
            yield outlet


class Pipelines:
    Function = PipelineFunction
    Generator = PipelineGenerator

