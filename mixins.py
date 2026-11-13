# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024
@name:   Mixins Object
@author: Jack Kirby Cook
@file:   support/mixins.py

"""

from abc import ABC

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Mixin"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Mixin(ABC):
    def __init_subclass__(cls, **kwargs):
        try: super().__init_subclass__(**kwargs)
        except TypeError: super().__init_subclass__()

    def __new__(cls, *args, **kwargs):
        try: return super().__new__(cls, *args, **kwargs)
        except TypeError: return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        try: super().__init__(*args, **kwargs)
        except TypeError: super().__init__()


