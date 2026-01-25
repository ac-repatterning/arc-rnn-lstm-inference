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

    def __init__(self, attribute: atr.Attribute, arguments: dict):
        """

        :param attribute: The attributes of a model and its supplements.  Refer to src.elements.attribute.py<br>
        :param arguments: A set of arguments vis-à-vis computation & storage objectives.<br>
        """

        self.__attribute = attribute
        self.__arguments = arguments

        # from attribute
        self.__modelling: dict = attribute.modelling

        # Renaming
        self.__rename = { arg: f'e_{arg}' for arg in self.__modelling.get('targets')}

    def __get_structure(self, frame: pd.DataFrame) -> pd.DataFrame:
        """

        :param frame:
        :return:
        """

        # Note, the unit of measure of `timestamps` must be milliseconds.
        dates = pd.date_range(start=frame['date'].max(), periods=self.__attribute.n_points_future + 1,
                              freq=self.__arguments.get('frequency'), inclusive='right', unit='ms')
        timestamps = dates.astype(np.int64)
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

        # The forecasts template
        template = structure.copy()

        # Temporary
        history = initial.copy()
        for _ in range(self.__attribute.n_points_future):
            value = model(history[:, -self.__modelling.get('n_sequence'):, :])
            history = np.concatenate((history, [[[float(value.numpy().squeeze())]]]), axis=1)
        template.loc[:, self.__modelling.get('targets')] = history[:, -self.__attribute.n_points_future:, :].squeeze()

        return template.copy()

    def __reconfigure(self, data: pd.DataFrame) -> pd.DataFrame:
        """

        :param data:
        :return:
        """

        instances = src.inference.scaling.Scaling().inverse_transform(
            data=data, scaling=self.__attribute.scaling)
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
        past = frame.copy()[-self.__modelling.get('n_sequence'):]
        structure = self.__get_structure(frame=frame)

        # Forecasting
        __future = self.__forecasting(model=model, past=past, structure=structure)
        future = self.__reconfigure(data=__future.copy())
        future.loc[:, 'ts_id'] = specification.ts_id
        future.loc[:, 'measure'] = np.nan

        return future
