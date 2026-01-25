"""Module cases.py"""
import os

import dask
import numpy as np
import pandas as pd

import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.s3.prefix


class Cases:
    """
    Retrieves the catchment & time series codes of the gauges in focus.
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param service: A suite of services for interacting with Amazon Web Services.<br>
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.<br>
        :param arguments: A set of arguments vis-à-vis computation & storage objectives.
        """

        self.__service = service
        self.__s3_parameters = s3_parameters
        self.__arguments = arguments

        # An instance for interacting with objects within an Amazon S3 prefix
        self.__pre = src.s3.prefix.Prefix(service=self.__service,
                                          bucket_name=self.__s3_parameters.internal)

    @staticmethod
    def __get_elements(objects: list[str]) -> pd.DataFrame:
        """

        :param objects:
        :return:
        """

        # A set of S3 uniform resource locators
        values = pd.DataFrame(data={'uri': objects})
        values = values.assign(uri=values['uri'].apply(os.path.dirname))
        values.drop_duplicates(inplace=True, ignore_index=True)

        # Splitting locators
        rename = {0: 'endpoint', 1: 'catchment_id', 2: 'ts_id'}
        splittings = values['uri'].str.rsplit('/', n=2, expand=True)
        splittings.rename(columns=rename, inplace=True)

        # Collating
        values = values.copy().join(splittings, how='left')

        # Drop 'endpoint'
        values.drop(columns='endpoint', inplace=True)

        return values

    def __get_cases(self, keys: list[str]) -> pd.DataFrame:
        """

        :param keys:
        :return:
        """

        # ... ensure the core model directory is excluded
        if len(keys) > 0:
            objects = [f's3://{self.__s3_parameters.internal}/{key}' for key in keys
                       if os.path.basename(os.path.dirname(key)) != 'model']
        else:
            return pd.DataFrame()

        # The variable objects is a list of uniform resource locators.  Each locator includes a 'ts_id',
        # 'catchment_id', 'datestr' substring; the function __get_elements extracts these items.
        values = self.__get_elements(objects=objects)

        # Types
        values['catchment_id'] = values['catchment_id'].astype(dtype=np.int64)
        values['ts_id'] = values['ts_id'].astype(dtype=np.int64)

        return values

    @dask.delayed
    def __get_listings(self, path: str) -> list[str]:
        """

        :param path:
        :return:
        """

        listings = self.__pre.objects(prefix=path, delimiter='')

        return listings

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        stage = self.__arguments.get('prefix').get('model')
        paths = self.__pre.objects(
            prefix=self.__arguments.get('modelling').get('path').get(stage), delimiter='/')
        paths = self.__pre.objects(paths[0], delimiter='/')

        computations = []
        for path in paths:
            computations.append(self.__get_listings(path=path))
        elements = dask.compute(computations, scheduler='threads')[0]
        keys: list[str] = sum(elements, [])

        return self.__get_cases(keys=keys)
