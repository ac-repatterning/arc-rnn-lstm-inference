"""Module approximating.py"""
import os

import pandas as pd
import tensorflow as tf

import config
import src.elements.approximations as apr
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

        :param arguments: A set of arguments vis-à-vis computation & storage objectives.<br>
        """

        self.__arguments = arguments

        # configurations
        self.__configurations = config.Config()

    def __get_model(self, specification: sc.Specification):
        """

        :param specification: Refer to src.elements.specification.py
        :return:
        """

        path = os.path.join(
            self.__configurations.data_, 'artefacts', str(specification.catchment_id), str(specification.ts_id))

        return tf.keras.models.load_model(
            filepath=os.path.join(path, 'model'))

    def exc(self, specification: sc.Specification, attribute: atr.Attribute, master: mr.Master) -> apr.Approximations:
        """

        :param specification: Refer to src.elements.specification.py
        :param attribute: Refer to src.elements.attribute.py
        :param master: Refer to src.elements.master.py
        :return:
        """

        model = self.__get_model(specification=specification)

        estimates: pd.DataFrame = src.inference.estimate.Estimate(attribute=attribute).exc(
            model=model, master=master)
        forecasts: pd.DataFrame = src.inference.forecast.Forecast(attribute=attribute, arguments=self.__arguments).exc(
            model=model, master=master, specification=specification)

        return apr.Approximations(estimates=estimates, forecasts=forecasts)
