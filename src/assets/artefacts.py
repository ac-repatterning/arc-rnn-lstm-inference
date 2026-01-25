"""Module artefacts.py"""
import logging
import os

import dask

import config
import src.elements.s3_parameters as s3p
import src.elements.specification as sc
import src.functions.directories
import src.s3.directives


class Artefacts:
    """
    Unloads artefacts; for inference.
    """

    def __init__(self, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param s3_parameters: The overarching S3 parameters settings of this project, e.g., region code
                              name, buckets, etc.
        :param arguments:
        """

        self.__s3_parameters = s3_parameters
        self.__arguments = arguments

        self.__configurations = config.Config()
        self.__directives =  src.s3.directives.Directives()
        self.__directories = src.functions.directories.Directories()

    def __acquire(self, specification: sc.Specification):
        """

        :param specification: Refer to src.elements.specification.py
        :return:
        """

        stage = self.__arguments.get('prefix').get('model')
        origin = (f'{self.__arguments.get('modelling').get('path').get(stage)}/'
                  f'{specification.catchment_id}/{specification.ts_id}')
        target = os.path.join(
            self.__configurations.data_, 'artefacts', str(specification.catchment_id), str(specification.ts_id))

        self.__directories.create(target)

        return self.__directives.unload(
            source_bucket=self.__s3_parameters.internal, origin=origin, target=target)

    def exc(self, specifications: list[sc.Specification]):
        """

        :param specifications: A list items of type Specification; refer to src.elements.specification.py
        :return:
        """

        # Or
        computations = [dask.delayed(self.__acquire)(specification=specification)
                        for specification in specifications]

        # Compute
        states = dask.compute(computations, scheduler='threads')[0]
        logging.info(states)
