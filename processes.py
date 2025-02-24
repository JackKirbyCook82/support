# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 2025
@name:   Process Objects
@author: Jack Kirby Cook

"""

from abc import ABC, ABCMeta, abstractmethod

from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Process"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ProcessMeta(ABCMeta):
    pass


class Process(Logging, ABC, metaclass=ProcessMeta):
    @abstractmethod
    def execute(self, *args, **kwargs): pass



