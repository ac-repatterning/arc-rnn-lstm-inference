"""Module specific.py"""
import argparse
import logging
import sys

import src.functions.cache


class Specific:
    """
    Specific
    """

    def __init__(self):
        """
        Constructor
        """

        self.__cache = src.functions.cache.Cache()

    @staticmethod
    def codes(value: str=None) -> list[int] | None:
        """

        :param value:
        :return:
        """

        if value is None:
            return None

        # Split and strip
        elements = [e.strip() for e in value.split(',')]

        try:
            _codes = [int(element) for element in elements]
        except argparse.ArgumentTypeError as err:
            raise err from err

        return _codes

    def request(self, value: str='0') -> int:
        """

        :param value:
        :return:
        """

        try:
            _value = int(value)
        except argparse.ArgumentTypeError as err:
            logging.info(('The optional parameter --request expects an integer; '
                          '0 indicates pre-live latest models, 1 indicates live publication of latest models, '
                          '2 indicates on-demand inference service, 3 indicates warning period inference.'))
            self.__cache.exc()
            raise err from err

        if _value in {0, 1, 2, 3}:
            return _value

        self.__cache.exc()
        sys.exit(('The optional parameter --request expects an integer; '
                  '0 indicates pre-live latest models, 1 indicates live publication of latest models, '
                  '2 indicates on-demand inference service, 3 indicates warning period inference.'))
