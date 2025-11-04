"""Module forecast.py"""
import numpy as np
import pandas as pd
import tensorflow as tf

import src.elements.attribute as atr
import src.elements.master as mr


class Forecast:
    """
    Under Development
    """

    def __init__(self, arguments: dict, attribute: atr.Attribute):
        """

        :param arguments:
        :param attribute:
        """


        self.__attribute = attribute
        self.__arguments = arguments

        # ...
        self.__n_points_future = self.__arguments.get('n_points_future')
        self.__cutoff = self.__arguments.get('n_points_future') + self.__attribute.modelling.get('n_sequence')

    def __get_structure(self, frame: pd.DataFrame) -> pd.DataFrame:
        """

        :param frame:
        :return:
        """

        dates = pd.date_range(start=frame['date'].max(), periods=self.__n_points_future+1, freq='h', inclusive='right')
        timestamps = (dates.astype(np.int64) / (10 ** 6)).astype(np.longlong)
        tail = pd.DataFrame(data={'timestamp': timestamps, 'date': dates})

        for i in self.__attribute.modelling.get('targets'):
            tail.loc[:, i] = np.nan

        __structure = pd.concat([frame.copy(), tail], axis=0, ignore_index=True)

        return __structure.copy()[-self.__cutoff:]

    def exc(self, model: tf.keras.models.Sequential, master: mr.Master):
        """

        :param model:
        :param master:
        :return:
        """

        frame = master.transforms
        structure = self.__get_structure(frame=frame)
