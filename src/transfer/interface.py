"""Module interface.py"""
import logging

import boto3
import pandas as pd

import config
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.s3.ingress
import src.transfer.cloud
import src.transfer.dictionary
import src.transfer.metadata


class Interface:
    """
    Class Interface
    """

    def __init__(self, connector: boto3.session.Session, service: sr.Service, s3_parameters: s3p, arguments: dict):
        """

        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.<br>
        :param service: A suite of services for interacting with Amazon Web Services. <br>
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.<br>
        :param arguments: A set of arguments vis-à-vis computation & storage objectives.<br>
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__arguments = arguments

        # Configurations
        self.__configurations = config.Config()

        # Metadata dictionary
        self.__metadata = src.transfer.metadata.Metadata(connector=connector)

    def __get_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        """

        :param frame:
        :return:
        """

        _metadata = self.__metadata.exc(name='metadata.json')

        frame = frame.assign(
            metadata = frame['section'].map(lambda x: _metadata[x]))

        return frame

    def __transfer(self, strings: pd.DataFrame):
        """

        :param strings: A table outlining the files to be transferred
        :return:
        """

        # Add metadata field
        strings = self.__get_metadata(frame=strings.copy())

        # Depending on the request, clear the targetted storage area first
        if self.__arguments.get('request') in {0, 3}:
            src.transfer.cloud.Cloud(
                service=self.__service, s3_parameters=self.__s3_parameters, arguments=self.__arguments).exc()

        # Finally, transfer
        messages = src.s3.ingress.Ingress(
            service=self.__service, bucket_name=self.__s3_parameters.external).exc(
            strings=strings, tags={'project': self.__configurations.project_tag})
        logging.info(messages)

    def exc(self):
        """

        :return:
        """

        # The strings for transferring data to Amazon S3 (Simple Storage Service)
        strings: pd.DataFrame = src.transfer.dictionary.Dictionary().exc(
            path=self.__configurations.pathway_, extension='*',
            prefix=self.__arguments.get('prefix').get('destination') + '/')

        if strings.empty:
            logging.info('There are no inference artefacts to transfer.')
        else:
            self.__transfer(strings=strings)
