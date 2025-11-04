"""Module scaling.py"""

import numpy as np
import pandas as pd


class Scaling:
    """
    Scaling
    """

    def __init__(self):
        pass

    @staticmethod
    def inverse_transform(data: pd.DataFrame, scaling: dict):
        """

        :param data:
        :param scaling:
        :return:
        """

        # The values
        structure: np.ndarray = data.copy()[scaling.get('feature_names_in_')].values

        # The inverse transform of ...
        matrix = np.multiply(structure, scaling.get('data_range_')) + scaling.get('data_min_')

        # Hence
        frame = pd.DataFrame()
        frame.loc[:, scaling.get('feature_names_in_')] = matrix

        return frame

    @staticmethod
    def transform(data: pd.DataFrame, scaling: dict) -> pd.DataFrame:
        """

        :param data:
        :param scaling:
        :return:
        """

        frame = data.copy()
        values = frame[scaling.get('feature_names_in_')].values
        matrix = np.true_divide(values - scaling.get('data_min_'), scaling.get('data_range_'))
        frame.loc[:, scaling.get('feature_names_in_')] = matrix

        return frame
