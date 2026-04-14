# -*- coding: utf-8 -*-
"""
Created on Tues Apr 14 2026
@name:   Filter Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC

from support.equations import Equations
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Filter(Equations, Logging, ABC):
    def filter(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty): return dataframe
        mask = self.equate(dataframe, *args, **kwargs).squeeze()
        dataframe = dataframe.where(mask)
        dataframe = dataframe.dropna(how="all", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe



