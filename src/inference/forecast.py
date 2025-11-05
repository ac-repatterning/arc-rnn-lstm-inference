"""Module forecast.py"""
import numpy as np
import pandas as pd
import tensorflow as tf

import src.elements.attribute as atr
import src.elements.master as mr
import src.inference.scaling


class Forecast:
    """
    Under Development
    """

    def __init__(self, attribute: atr.Attribute):
        """

        :param attribute:
        """

        self.__modelling = attribute.modelling
        self.__scaling = attribute.scaling
        self.__n_points_future = attribute.n_points_future

        # And
        self.__n_sequence = self.__modelling.get('n_sequence')

        # Renaming
        self.__rename = { arg: f'e_{arg}' for arg in self.__modelling.get('targets')}

    def __get_structure(self, frame: pd.DataFrame) -> pd.DataFrame:
        """

        :param frame:
        :return:
        """

        dates = pd.date_range(start=frame['date'].max(), periods=self.__n_points_future+1, freq='h', inclusive='right')
        timestamps = (dates.astype(np.int64) / (10 ** 6)).astype(np.longlong)
        future = pd.DataFrame(data={'timestamp': timestamps, 'date': dates})

        for i in self.__modelling.get('targets'):
            future.loc[:, i] = np.nan

        return future

    # pylint: disable=E1101
    def __forecasting(self, model: tf.keras.models.Sequential, past: pd.DataFrame, f_structure: pd.DataFrame) -> pd.DataFrame:
        """

        :param model:
        :param past:
        :param f_structure:
        :return:
        """

        # History
        initial = past[self.__modelling.get('fields')].values[None, :]
        history = initial.copy()

        # The forecasts template
        template = f_structure.copy()

        # Hence
        for i in range(self.__n_points_future):
            values = model.predict(x=history[:, -self.__n_sequence:, :], verbose=0)
            template.loc[i, self.__modelling.get('targets')] = values
            affix = template.loc[i, self.__modelling.get('fields')].values.astype(float)
            history = np.concatenate((history, affix[None, None, :]), axis=1)

        return template.copy()

    def __reconfigure(self, data: pd.DataFrame) -> pd.DataFrame:
        """

        :param data:
        :return:
        """

        frame = src.inference.scaling.Scaling().inverse_transform(
            data=data, scaling=self.__scaling)

        return frame.rename(columns=self.__rename, inplace=True)

    # pylint: disable=E1101
    def exc(self, model: tf.keras.models.Sequential, master: mr.Master) -> pd.DataFrame:
        """

        :param model:
        :param master:
        :return:
        """

        # The frame that has the scaled fields
        frame = master.transforms

        # Predicting future values requires (a) past values, and (b) a structure for future values
        past = frame.copy()[-self.__n_sequence:]
        f_structure = self.__get_structure(frame=frame)

        # Forecasting
        __future = self.__forecasting(model=model, past=past, f_structure=f_structure)
        future = self.__reconfigure(data=__future.copy())

        return future
