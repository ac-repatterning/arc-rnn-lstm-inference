import logging
import sys

import pandas as pd

import src.elements.service as sr
import src.elements.s3_parameters as s3p

import src.assets.cases
import src.assets.filtering
import src.assets.foci
import src.functions.cache


class Metadata:

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param service:
        :param s3_parameters: The overarching S3 (Simple Storage Service) parameters
                              settings of this project, e.g., region code name, buckets, etc.
        :param arguments:
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__arguments = arguments

    def __get_metadata(self, cases: pd.DataFrame, foci: pd.DataFrame) -> pd.DataFrame:
        """

        :param cases:
        :param foci:
        :return:
        """

        # filter in relation to context - live, on demand via input argument, inspecting inference per model
        metadata = src.assets.filtering.Filtering(
            cases=cases.copy(), foci=foci.copy(), arguments=self.__arguments).exc()

        if metadata.empty:
            logging.info('Nothing to do.  Is your inference request in relation to one or more existing models?')
            src.functions.cache.Cache().exc()
            sys.exit(0)

        return metadata

    def exc(self):
        """

        :return:
        """

        # the identification codes of gauge stations vis-à-vis existing model artefacts
        cases = src.assets.cases.Cases(
            service=self.__service, s3_parameters=self.__s3_parameters, arguments=self.__arguments).exc()

        # gauge stations identifiers vis-à-vis warning period
        foci = src.assets.foci.Foci(s3_parameters=self.__s3_parameters).exc()

        return self.__get_metadata(cases=cases, foci=foci)
