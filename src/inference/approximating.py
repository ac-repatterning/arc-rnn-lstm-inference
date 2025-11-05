"""Module approximating.py"""
import logging
import os

import tensorflow as tf

import config
import src.elements.attribute as atr
import src.elements.master as mr
import src.elements.specification as sc
import src.inference.estimate
import src.inference.forecast


class Approximating:
    """
    Under Development
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments

        # Instances
        self.__configurations = config.Config()

    def __get_model(self, specification: sc.Specification) -> tf.keras.models.Sequential:
        """

        :param path:
        :return:
        """

        path = os.path.join(self.__configurations.data_, str(specification.catchment_id), str(specification.ts_id))

        return tf.keras.models.load_model(
            filepath=os.path.join(path, 'model.keras'))

    def exc(self, specification: sc.Specification, attribute: atr.Attribute, master: mr.Master):
        """

        :param specification:
        :param attribute:
        :param master:
        :return:
        """

        # Read-in the gauge's model
        model = self.__get_model(specification=specification)

        # Subsequently, estimate w.r.t. to the known, and forecast w.r.t. the unknown.
        estimates = src.inference.estimate.Estimate(attribute=attribute).exc(model=model, master=master)
        forecasts = src.inference.forecast.Forecast(attribute=attribute).exc(model=model, master=master)
        logging.info(estimates)
        logging.info(forecasts)
