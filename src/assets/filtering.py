"""Module filtering.py"""
import numpy as np
import pandas as pd


class Filtering:
    """
    Filtering
    """

    def __init__(self, cases: pd.DataFrame, arguments: dict):
        """

        :param cases:
        :param arguments:
        """

        self.__cases = cases
        self.__arguments = arguments

    def __filtering(self) -> pd.DataFrame:
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

        return self.__filtering()
