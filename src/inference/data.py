"""Module data.py"""

import dask.dataframe as ddf
import numpy as np
import pandas as pd


class Data:
    """
    Data
    """

    def __init__(self):
        """
        Constructor
        """

        # Focus
        self.__dtype = {'timestamp': np.float64, 'ts_id': np.float64, 'measure': float}


    def __get_data(self, listing: list[str]):
        """

        :param listing:
        :return:
        """

        try:
            block: pd.DataFrame = ddf.read_csv(
                listing, header=0, usecols=list(self.__dtype.keys()), dtype=self.__dtype).compute()
        except ImportError as err:
            raise err from err

        block.reset_index(drop=True, inplace=True)
        block.sort_values(by='timestamp', ascending=True, inplace=True)
        block.drop_duplicates(subset='timestamp', keep='first', inplace=True)

        return block

    @staticmethod
    def __set_missing(data: pd.DataFrame) -> pd.DataFrame:
        """
        Forward filling.  In contrast, the variational model inherently deals with missing data, hence
                          it does not include this type of step.

        :param data:
        :return:
        """

        data['measure'] = data['measure'].ffill().values

        return data

    def exc(self, listing: list[str], modelling: dict) -> pd.DataFrame:
        """
        Append a date of the format datetime64[]
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

        :param listing:
        :param modelling: A set of modelling stage arguments
        :return:
        """

        # The data
        data = self.__get_data(listing=listing)
        data = self.__set_missing(data=data.copy())

        # Filter
        __as_from = modelling.get('training_starts').get('epoch_milliseconds')
        data = data.copy().loc[data['timestamp'] >= __as_from, :]

        return data
