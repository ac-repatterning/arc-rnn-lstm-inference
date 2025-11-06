"""Module data.py"""
import os

import dask.dataframe as ddf
import numpy as np
import pandas as pd

import config
import src.elements.attribute as atr
import src.elements.specification as sc


class Data:
    """
    Data
    """

    def __init__(self, limits: list):
        """

        :param limits: The list of data-files-dates in focus.
        """

        self.__limits = limits

        # Configurations
        self.__configurations = config.Config()

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

    def __get_listing(self, specification: sc.Specification) -> list[str]:
        """

        :param specification:
        :return:
        """

        listing  = [os.path.join(self.__configurations.data_, 'source', str(specification.catchment_id),
                                 str(specification.ts_id), f'{limit}.csv' )
                    for limit in self.__limits]

        return listing

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

    def exc(self, specification: sc.Specification, attribute: atr.Attribute) -> pd.DataFrame:
        """
        __as_from = attribute.modelling.get('training_starts').get('epoch_milliseconds')
        data = data.copy().loc[data['timestamp'] >= __as_from, :]

        :param specification:
        :param attribute:
        :return:
        """

        listing =  self.__get_listing(specification=specification)

        # The data
        data = self.__get_data(listing=listing)
        data = self.__set_missing(data=data.copy())

        # Filter
        n_samples_seen_ = attribute.scaling.get('n_samples_seen_')
        data = data.copy().loc[-n_samples_seen_:, :]

        # datetime
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

        return data
