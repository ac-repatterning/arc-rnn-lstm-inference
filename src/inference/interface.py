"""Module inference/interface.py"""
import logging

import boto3
import dask

import src.elements.attribute as atr
import src.elements.specification as sc
import src.inference.attributes


class Interface:
    """
    Interface
    """

    def __init__(self, connector: boto3.session.Session, limits: list):
        """

        :param connector:
        :param limits:
        """

        self.__connector = connector
        self.__limits = limits

    def exc(self, specifications: list[sc.Specification]):
        """

        :param specifications:
        :return:
        """

        __get_attributes = dask.delayed(src.inference.attributes.Attributes(connector=self.__connector).exc)

        computations = []
        for specification in specifications:
            attribute: atr.Attribute = __get_attributes(specification=specification)
            computations.append(attribute.modelling['n_sequence'])

        calculations = dask.compute(computations, scheduler='threads')[0]
        logging.info(calculations)
