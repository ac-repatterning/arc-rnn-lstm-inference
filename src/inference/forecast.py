"""Module forecast.py"""
import numpy as np
import pandas as pd
import tensorflow as tf

import src.elements.attribute as atr
import src.elements.master as mr
import src.elements.specification as sc
import src.inference.scaling


class Forecast:
    """
    For forecasting vis-à-vis the future
    """

    def __init__(self, attribute: atr.Attribute):
        """

        :param attribute: Refer to src.elements.attribute.py
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
        structure = pd.DataFrame(data={'timestamp': timestamps, 'date': dates})

        for i in self.__modelling.get('targets'):
            structure.loc[:, i] = np.nan

        return structure

    # pylint: disable=E1101
    def __forecasting(self, model: tf.keras.models.Sequential, past: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
        """

        :param model:
        :param past:
        :param structure:
        :return:
        """

        # History
        initial = past[self.__modelling.get('fields')].values[None, :]
        history = initial.copy()

        # The forecasts template
        template = structure.copy()

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

        instances = src.inference.scaling.Scaling().inverse_transform(
            data=data, scaling=self.__scaling)
        instances = instances.copy().rename(columns=self.__rename)

        frame = data.copy()
        frame = frame.copy().drop(columns=self.__modelling.get('targets'))
        frame.loc[:, list(self.__rename.values())] = instances.values

        return frame

    # pylint: disable=E1101
    def exc(self, model: tf.keras.models.Sequential, master: mr.Master, specification: sc.Specification) -> pd.DataFrame:
        """

        :param model:
        :param master:
        :param specification
        :return:
        """

        # The frame that has the scaled fields
        frame = master.transforms

        # Predicting future values requires (a) past values, and (b) a structure for future values
        past = frame.copy()[-self.__n_sequence:]
        structure = self.__get_structure(frame=frame)

        # Forecasting
        __future = self.__forecasting(model=model, past=past, structure=structure)
        future = self.__reconfigure(data=__future.copy())
        future.loc[:, 'ts_id'] = specification.ts_id
        future.loc[:, 'measure'] = np.nan

        return future
