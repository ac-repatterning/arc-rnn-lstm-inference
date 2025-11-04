"""Module scaling.py"""

import numpy as np
import pandas as pd


class Scaling:
    """
    Scaling
    """

    def __init__(self):
        pass

    def inverse_transform(self):
        pass

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
