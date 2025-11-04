"""Module artefacts.py"""
import logging
import dask
import os

import config
import src.elements.specification as sc
import src.elements.s3_parameters as s3p
import src.functions.directories
import src.s3.directives

class Artefacts:
    """
    Unloads artefacts; for inference.
    """

    def __init__(self, s3_parameters: s3p.S3Parameters):
        """

        :param s3_parameters:
        """

        self.__s3_parameters = s3_parameters

        self.__configurations = config.Config()
        self.__directives =  src.s3.directives.Directives()
        self.__directories = src.functions.directories.Directories()

    @dask.delayed
    def __acquire(self, specification: sc.Specification):
        """

        :param specification:
        :return:
        """

        origin = f'{self.__configurations.artefacts_}/{specification.catchment_id}/{specification.ts_id}'
        target = os.path.join(self.__configurations.data_, str(specification.catchment_id), str(specification.ts_id))

        self.__directories.create(target)

        return self.__directives.unload(
            source_bucket=self.__s3_parameters.internal, origin=origin, target=target)

    def exc(self, specifications: list[sc.Specification]):
        """

        :param specifications:
        :return:
        """

        computations = []
        for specification in specifications:
            state = self.__acquire(specification=specification)
            computations.append(state)

        states = dask.compute(computations, scheduler='threads')[0]
        logging.info(states)
