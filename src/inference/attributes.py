"""Module inference/attributes.py"""
import logging
import os

import config
import src.elements.attribute as atr
import src.elements.specification as sc
import src.functions.objects
import src.s3.unload


class Attributes:
    """
    Attributes
    """

    def __init__(self):
        """
        Constructor
        """

        # Instances
        self.__configurations = config.Config()
        self.__objects = src.functions.objects.Objects()

    def __get_request(self, uri: str) -> dict | list[dict]:
        """

        :param uri: A file's uniform resource identifier.
        :return:
        """

        return self.__objects.read(uri=uri)

    def exc(self, specification: sc.Specification) -> atr.Attribute:
        """

        :param specification: Refer to src.elements.specification.py
        :return:
        """

        path = os.path.join(self.__configurations.data_, str(specification.catchment_id), str(specification.ts_id))

        attribute = atr.Attribute(
            modelling=self.__get_request(uri=os.path.join(path, 'modelling.json')),
            scaling=self.__get_request(uri=os.path.join(path, 'scaling.json')))

        logging.info(attribute.modelling)
        logging.info(attribute.scaling)

        return attribute
