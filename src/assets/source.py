"""Module source.py"""
import logging
import os

import dask

import config
import src.elements.specification as sc
import src.s3.directives


class Source:
    """
    Source
    """

    def __init__(self, arguments: dict, limits: list):
        """

        :param arguments:
        :param limits:
        """

        self.__arguments = arguments
        self.__limits = limits

        # Endpoint
        self.__endpoint: str = self.__arguments.get('additions').get('modelling_data_source')

        # Instances
        self.__configurations = config.Config()
        self.__directives =  src.s3.directives.Directives()

    @dask.delayed
    def __acquire(self, specification: sc.Specification) -> int:
        """

        :param specification: Refer to src.elements.specification.py
        :return:
        """

        # Focusing on the relevant data sets
        parts = [ f"--include \'{limit}*\'" for limit in self.__limits]
        extra = '--recursive ' + "--exclude \'*\' " + ' '.join(parts)

        # key & target
        key = f'{self.__endpoint}/{specification.catchment_id}/{specification.ts_id}/'
        target = os.path.join(
            self.__configurations.data_, 'source', str(specification.catchment_id), str(specification.ts_id))

        status = self.__directives.unload_(key=key, target=target, extra=extra)

        return status

    def exc(self, specifications: list[sc.Specification]) -> None:
        """

        :param specifications: A list items of type Specification; refer to src.elements.specification.py
        :return:
        """

        computations = []
        for specification in specifications:
            state = self.__acquire(specification=specification)
            computations.append(state)

        states = dask.compute(computations, scheduler='threads')[0]
        logging.info(states)
