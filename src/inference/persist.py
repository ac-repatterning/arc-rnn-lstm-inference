import logging
import json
import os
import pandas as pd

import config
import src.elements.approximations as apr
import src.elements.specification as sc
import src.functions.objects


class Persist:

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

        # An instance for writing JSON objects
        self.__objects = src.functions.objects.Objects()

    @staticmethod
    def __get_node(blob: pd.DataFrame) -> dict:
        """

        :param blob:
        :return:
        """

        string: str = blob.to_json(orient='split')

        return json.loads(string)

    def __persist(self, nodes: dict, name: str) -> str:
        """

        :param nodes: Dictionary of data.
        :param name: A name for the file.
        :return:
        """

        return self.__objects.write(
            nodes=nodes, path=os.path.join(self.__configurations.points_, f'{name}.json'))

    def exc(self, specification: sc.Specification, approximations: apr.Approximations) -> str:
        """
        * Add an absolute percentage error field to the `approximations.estimates` frame<br>
        * Per `approximations` frame drop `date` & `ts_id`<br><br>

        :param specification: <br>
        :param approximations: <br>
        :return:
        """

        nodes = {
            'estimates': self.__get_node(approximations.estimates.drop(columns=['date', 'ts_id'])),
            'forecasts': self.__get_node(approximations.forecasts.drop(columns=['date', 'ts_id']))
        }
        nodes.update(specification._asdict())

        return self.__persist(nodes=nodes, name=str(specification.ts_id))
