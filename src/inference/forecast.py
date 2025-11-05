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

        self.__modelling = attribute.modelling
        self.__scaling = attribute.scaling
        self.__arguments = arguments

        # ...
        self.__n_points_future = self.__arguments.get('n_points_future')
        self.__n_sequence = self.__modelling.get('n_sequence')

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

    def __forecasting(self, model: tf.keras.models.Sequential, past: pd.DataFrame, future: pd.DataFrame):

        # History
        initial = past[self.__modelling.get('fields')].values[None, :]
        history = initial.copy()

        # The forecasts template
        template = future.copy()

        # Hence
        for i in range(self.__n_points_future):
            values = model.predict(x=history[:, -self.__n_sequence:, :], verbose=0)
            template.loc[i, self.__modelling.get('targets')] = values
            affix = template.loc[i, self.__modelling.get('fields')].values.astype(float)
            history = np.concatenate((history, affix[None, None, :]), axis=1)

        return template.copy()

    def exc(self, model: tf.keras.models.Sequential, master: mr.Master):
        """

        :param model:
        :param master:
        :return:
        """

        frame = master.transforms

        past = frame.copy()[-self.__n_sequence:]
        future = self.__get_structure(frame=frame)
        self.__forecasting(model=model, past=past, future=future)
