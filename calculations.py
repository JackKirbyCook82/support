# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import logging
from abc import ABC, ABCMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class CalculationMeta(ABCMeta):
    pass


class Calculation(ABC, metaclass=CalculationMeta):
    pass



