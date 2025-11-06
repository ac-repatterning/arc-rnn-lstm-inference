"""Module approximating.py"""
import os

import pandas as pd
import tensorflow as tf

import config

import src.elements.specification as sc
import src.elements.attribute as atr
import src.elements.master as mr
import src.inference.estimate
import src.inference.forecast


class Approximating:
    """
    Under Development
    """

    def __init__(self):
        """
        Constructor
        """

        # Instances
        self.__configurations = config.Config()

    def __get_model(self, specification: sc.Specification):
        """


        :param specification:
        :return:
        """

        path = os.path.join(
            self.__configurations.data_, 'artefacts', str(specification.catchment_id), str(specification.ts_id))

        return tf.keras.models.load_model(
            filepath=os.path.join(path, 'model.keras'))

    def exc(self, specification: sc.Specification, attribute: atr.Attribute, master: mr.Master):

        model = self.__get_model(specification=specification)

        estimates: pd.DataFrame = src.inference.estimate.Estimate(attribute=attribute).exc(model=model, master=master)
        forecasts: pd.DataFrame = src.inference.forecast.Forecast(attribute=attribute).exc(model=model, master=master)
        frame = pd.concat([estimates, forecasts], ignore_index=True, axis=0)

        return frame.shape[0]

