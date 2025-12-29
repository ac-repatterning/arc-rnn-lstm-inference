"""Module interface.py"""

import pandas as pd

import src.assets.artefacts
import src.assets.menu
import src.assets.metadata
import src.assets.reference
import src.assets.source
import src.assets.specifications
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.elements.specification as sc
import src.functions.cache


class Interface:
    """
    Interface
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param service: A suite of services for interacting with Amazon Web Services.<br>
        :param s3_parameters: The overarching S3 (Simple Storage Service) parameters
                              settings of this project, e.g., region code name, buckets, etc.
        :param arguments: A set of arguments vis-à-vis computation & storage objectives.<br>
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__arguments = arguments

    def exc(self, limits: list) -> list[sc.Specification]:
        """

        :param limits:
        :return:
        """

        # The gauge stations in focus
        metadata = src.assets.metadata.Metadata(
            service=self.__service, s3_parameters=self.__s3_parameters, arguments=self.__arguments).exc()

        # Reference
        reference: pd.DataFrame = src.assets.reference.Reference(
            s3_parameters=self.__s3_parameters).exc(codes=metadata['ts_id'].unique())
        reference = reference.copy().merge(metadata, how='left', on=['catchment_id', 'ts_id'])

        # Menu
        src.assets.menu.Menu().exc(reference=reference)

        # Specifications
        specifications: list[sc.Specification] = src.assets.specifications.Specifications().exc(reference=reference)

        # Unload model artefacts
        src.assets.artefacts.Artefacts(
            s3_parameters=self.__s3_parameters, arguments=self.__arguments).exc(specifications=specifications)
        src.assets.source.Source(
            arguments=self.__arguments, limits=limits).exc(specifications=specifications)

        return specifications
