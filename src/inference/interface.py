"""Module inference/interface.py"""
import logging
import multiprocessing

import dask
import pandas as pd

import src.elements.approximations as apr
import src.elements.attribute as atr
import src.elements.master as mr
import src.elements.specification as sc
import src.inference.approximating
import src.inference.attributes
import src.inference.data
import src.inference.persist
import src.inference.scaling


class Interface:
    """
    Interface
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis computation & storage objectives.<br>
        """

        self.__arguments = arguments

        # Setting up
        self.__scaling = src.inference.scaling.Scaling()
        self.__n_cores = multiprocessing.cpu_count()

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
        __get_data = dask.delayed(src.inference.data.Data(arguments=self.__arguments).exc)
        __approximating = dask.delayed(src.inference.approximating.Approximating(arguments=self.__arguments).exc)
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

        messages = dask.compute(computations, scheduler='processes', num_workers=int(0.5*self.__n_cores))[0]
        logging.info(messages)
