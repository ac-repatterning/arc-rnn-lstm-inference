"""Module filtering.py"""
import logging
import sys

import numpy as np
import pandas as pd

import src.functions.cache


class Filtering:
    """
    Filtering
    """

    def __init__(self, cases: pd.DataFrame, foci: pd.DataFrame, arguments: dict):
        """

        :param cases: The gauge stations that have model artefacts.<br>
        :param foci: The gauge stations within a current warning area.<br>
        :param arguments: A set of arguments vis-à-vis computation & storage objectives.<br>
        """

        self.__cases = cases
        self.__foci = foci
        self.__arguments = arguments

    def __inspect(self) -> pd.DataFrame:
        """
        Beware, the number of cases herein will be due to the model artefacts that exist within the
        `inspect`, i.e., pre-live, storage area.

        :return:
        """

        values: list = self.__arguments.get('series').get('excerpt')

        if values is None:
            return self.__cases

        frame = self.__cases.copy().loc[self.__cases['ts_id'].isin(values), :]

        return frame

    def __miscellaneous(self) -> pd.DataFrame:
        """
        The number of cases herein will be due to the model artefacts that exist within the `live` storage area.

        :return:
        """

        __excerpt = self.__arguments.get('series').get('excerpt')
        if __excerpt is None:
            return pd.DataFrame()

        codes = np.unique(np.array(__excerpt))
        cases = self.__cases.copy().loc[self.__cases['ts_id'].isin(codes), :]
        cases = cases if cases.shape[0] > 0 else self.__cases

        return cases

    def __warning(self) -> pd.DataFrame:
        """

        :return:
        """

        if self.__foci.empty:
            return pd.DataFrame()

        codes: np.ndarray = self.__foci['ts_id'].values
        cases = self.__cases.copy().loc[self.__cases['ts_id'].isin(codes), :]
        cases = cases if cases.shape[0] > 0 else self.__cases

        return cases

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        match self.__arguments.get('request'):
            case 0:
                cases = self.__inspect()
            case 1:
                cases = self.__miscellaneous()
            case 2:
                cases = self.__miscellaneous()
            case 3:
                cases = self.__warning()
            case _:
                raise ValueError(f'Unknown request code: {self.__arguments.get('request')}')

        if cases.empty:
            logging.info('Nothing to do.  Is your inference request in relation to one or more existing models?')
            src.functions.cache.Cache().exc()
            sys.exit(0)

        return cases
