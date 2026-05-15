# -*- coding: utf-8 -*-
"""
Created on Tues Apr 14 2026
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from support.equations import Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Filter", "Generator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Filter(Equations, ABC):
    def filter(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty): return dataframe
        mask = self.execute(dataframe, *args, **kwargs)
        mask = mask.squeeze()
        dataframe = dataframe.where(mask)
        dataframe = dataframe.dropna(how="all", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class Generator(ABC):
    def generate(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty): return dataframe
        generator = self.generator(dataframe, *args, **kwargs)
        dataframes = list(generator)
        if bool(dataframes): dataframe = pd.concat(dataframes, axis=0)
        else: dataframe = pd.DataFrame(columns=dataframe.columns)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @abstractmethod
    def generator(self, dataframe, *args, **kwargs): pass



