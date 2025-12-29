"""Module metadata.py"""

import pandas as pd

import src.assets.cases
import src.assets.filtering
import src.assets.foci
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.cache


class Metadata:
    """
    Retrieves the metadata of the gauges in focus.
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param service: A suite of services for interacting with Amazon Web Services.<br>
        :param s3_parameters: The overarching S3 (Simple Storage Service) parameters
                              settings of this project, e.g., region code name, buckets, etc.<br>
        :param arguments: A set of arguments vis-à-vis computation & storage objectives.<br>
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__arguments = arguments

    def __get_metadata(self, cases: pd.DataFrame, foci: pd.DataFrame) -> pd.DataFrame:
        """

        :param cases: The gauge stations that have model artefacts.<br>
        :param foci: The gauge stations within a current warning area.<br>
        :return:
        """

        # filter in relation to context - live, on demand via input argument, inspecting inference per model
        metadata = src.assets.filtering.Filtering(
            cases=cases.copy(), foci=foci.copy(), arguments=self.__arguments).exc()

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
