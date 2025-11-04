"""Module interface.py"""
import typing

import boto3

import config
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.service
import src.preface.setup
import src.s3.configurations
import src.s3.s3_parameters


class Interface:
    """
    Interface
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

    @staticmethod
    def __set_source(arguments: dict, s3_parameters: s3p.S3Parameters) -> dict:
        """

        :param arguments:
        :param s3_parameters:
        :return:
        """

        objects = s3_parameters._asdict()
        bucket = objects[arguments.get('s3').get('p_bucket')]
        prefix = objects[arguments.get('s3').get('p_prefix')]

        source = f's3://{bucket}/{prefix}{arguments.get('s3').get('affix')}'

        arguments['additions'] = {'modelling_data_source': source}

        return arguments

    def __get_arguments(self, connector: boto3.session.Session) -> dict:
        """

        :param connector:
        :return:
        """

        key_name = self.__configurations.arguments_key

        arguments = src.s3.configurations.Configurations(connector=connector).objects(key_name=key_name)

        return arguments

    def exc(self, codes: list[int] | None) -> typing.Tuple[boto3.session.Session, s3p.S3Parameters, sr.Service, dict]:
        """

        :return:
        """

        connector = boto3.session.Session()

        # Interaction Instances: Amazon
        s3_parameters: s3p.S3Parameters = src.s3.s3_parameters.S3Parameters(connector=connector,).exc()
        service: sr.Service = src.functions.service.Service(
            connector=connector, region_name=s3_parameters.region_name).exc()

        # Arguments
        arguments: dict = self.__get_arguments(connector=connector)
        arguments: dict = self.__set_source(arguments=arguments.copy(), s3_parameters=s3_parameters)

        # Codes
        if codes is not None:
            arguments['series']['excerpt'] = codes

        src.preface.setup.Setup(service=service, s3_parameters=s3_parameters).exc()

        return connector, s3_parameters, service, arguments
