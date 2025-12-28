"""Module interface.py"""
import argparse
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

    def __get_arguments(self, connector: boto3.session.Session, s3_parameters: s3p.S3Parameters,
                        args: argparse.Namespace) -> dict:
        """

        :param connector:
        :param s3_parameters:
        :param args:
        :return:
        """

        key_name = self.__configurations.arguments_key
        arguments = src.s3.configurations.Configurations(connector=connector).objects(key_name=key_name)
        arguments: dict = self.__set_source(arguments=arguments.copy(), s3_parameters=s3_parameters)

        # Codes
        if args.codes is not None:
            arguments['series']['excerpt'] = args.codes
        else:
            arguments['series']['excerpt'] = []

        # Request
        arguments['request'] = args.request

        return arguments

    @staticmethod
    def __prefix(arguments: dict) -> dict:
        """

        :param arguments:
        :return:
        """

        match arguments.get('request'):
            case 0:
                arguments['prefix'] = arguments.get('inference').get('inspect')
            case 1:
                arguments['prefix'] = arguments.get('inference').get('live')
            case 2:
                arguments['prefix'] = arguments.get('inference').get('service')
            case 3:
                arguments['prefix'] = arguments.get('inference').get('warning')
            case _:
                raise ValueError(f'Unknown request code: {arguments.get('request')}')

        return arguments

    def exc(self, args: argparse.Namespace) -> typing.Tuple[boto3.session.Session, s3p.S3Parameters, sr.Service, dict]:
        """

        :param args: Wherein -> codes: list[int] | None, live: 0 | 1 | 2 | 3
        :return:
        """

        connector = boto3.session.Session()

        # Interaction Instances: Amazon
        s3_parameters: s3p.S3Parameters = src.s3.s3_parameters.S3Parameters(connector=connector,).exc()
        service: sr.Service = src.functions.service.Service(
            connector=connector, region_name=s3_parameters.region_name).exc()

        # Arguments
        arguments: dict = self.__get_arguments(connector=connector, s3_parameters=s3_parameters, args=args)
        arguments: dict = self.__prefix(arguments=arguments)

        # Setting up
        src.preface.setup.Setup(service=service, s3_parameters=s3_parameters).exc()

        return connector, s3_parameters, service, arguments
