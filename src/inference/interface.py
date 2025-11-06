"""Module inference/interface.py"""
import logging

import os
import dask
import pandas as pd

import config
import src.elements.attribute as atr
import src.elements.master as mr
import src.elements.specification as sc
import src.elements.approximations as apr
import src.inference.approximating
import src.inference.attributes
import src.inference.data
import src.inference.scaling
import src.inference.persist


class Interface:
    """
    Interface
    """

    def __init__(self, arguments: dict, limits: list):
        """

        :param arguments:
        :param limits:
        """

        self.__arguments = arguments
        self.__limits = limits

        # Setting up
        self.__configurations = config.Config()
        self.__endpoint: str = self.__arguments.get('additions').get('modelling_data_source')
        self.__scaling = src.inference.scaling.Scaling()

    @dask.delayed
    def __get_listing(self, specification: sc.Specification) -> list[str]:
        """

        :param specification:
        :return:
        """

        listing  = [os.path.join(self.__configurations.data_, 'source', str(specification.catchment_id), str(specification.ts_id), f'{limit}.csv' )
         for limit in self.__limits]

        # listing = [f'{self.__endpoint}/{specification.catchment_id}/{specification.ts_id}/{limit}.csv'
        #            for limit in self.__limits ]

        return listing

    @dask.delayed
    def __set_transforms(self, data: pd.DataFrame, scaling: dict) -> mr.Master:
        """

        :param data:
        :param scaling:
        :return:
        """

        transforms = self.__scaling.transform(data=data, scaling=scaling)

        return mr.Master(data=data, transforms=transforms)

    def exc(self, specifications: list[sc.Specification]):
        """

        :param specifications:
        :return:
        """

        __get_attributes = dask.delayed(src.inference.attributes.Attributes(arguments=self.__arguments).exc)
        __get_data = dask.delayed(src.inference.data.Data(limits=self.__limits).exc)
        __approximating = dask.delayed(src.inference.approximating.Approximating().exc)
        __persist = dask.delayed(src.inference.persist.Persist().exc)

        computations = []
        for specification in specifications:
            attribute: atr.Attribute = __get_attributes(specification=specification)
            data: pd.DataFrame = __get_data(specification=specification, attribute=attribute)
            master: mr.Master = self.__set_transforms(data=data, scaling=attribute.scaling)
            approximations: apr.Approximations = __approximating(
                specification=specification, attribute=attribute, master=master)
            message = __persist(specification=specification, approximations=approximations)
            computations.append(message)

        messages = dask.compute(computations, scheduler='threads')[0]
        logging.info(messages)
