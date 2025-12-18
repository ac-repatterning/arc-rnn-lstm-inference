"""Module filtering.py"""
import numpy as np
import pandas as pd


class Filtering:
    """
    Filtering
    """

    def __init__(self, cases: pd.DataFrame, foci: pd.DataFrame, arguments: dict):
        """

        :param cases: The gauge stations that have model artefacts
        :param foci: The gauge stations within a current warning area
        :param arguments:
        """

        self.__cases = cases
        self.__foci = foci
        self.__arguments = arguments

    def __live(self):
        """

        :return:
        """

        if self.__foci.empty:
            return pd.DataFrame()

        codes: np.ndarray = self.__foci['ts_id'].values
        cases = self.__cases.copy().loc[self.__cases['ts_id'].isin(codes)]
        cases = cases if cases.shape[0] > 0 else self.__cases

        return cases

    def __inherent(self) -> pd.DataFrame:
        """

        :return:
        """

        excerpt = self.__arguments.get('series').get('excerpt')
        if excerpt is None:
            cases =  self.__cases
        else:
            codes = np.unique(np.array(excerpt))
            cases = self.__cases.copy().loc[self.__cases['ts_id'].isin(codes), :]
            cases = cases if cases.shape[0] > 0 else self.__cases

        return cases

    def exc(self):
        """

        :return:
        """

        if self.__arguments.get('live'):
            return self.__live()

        return self.__inherent()
