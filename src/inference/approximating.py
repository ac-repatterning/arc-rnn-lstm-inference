"""Module approximating.py"""
import logging
import os

import tensorflow as tf

import config
import src.elements.specification as sc


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

    @staticmethod
    def __get_model(path: str) -> tf.keras.models.Sequential:
        """

        :param path:
        :return:
        """

        return tf.keras.models.load_model(
            filepath=os.path.join(path, 'model.keras'))

    def exc(self, specification: sc.Specification):
        """

        :param specification:
        :return:
        """

        logging.info(self.__arguments)

        path = os.path.join(self.__configurations.data_, str(specification.catchment_id), str(specification.ts_id))
        model = self.__get_model(path=path)
        logging.info(model)
