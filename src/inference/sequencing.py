"""Module sequencing.py"""
import typing

import numpy as np
import pandas as pd


class Sequencing:
    """
    Builds the modelling windows, i.e., sequences of historical data & corresponding target
    """

    def __init__(self, modelling: dict):
        """

        :param modelling: A set of modelling stage arguments
        """

        self.__n_sequence = modelling.get('n_sequence')
        self.__fields = modelling.get('fields')
        self.__targets = modelling.get('targets')

    def exc(self, blob: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
        """

        :param blob: A modelling data set
        :return:
        """

        data = blob.copy().loc[:, self.__fields]
        matrix = data.values

        # The indices of the target fields
        __indices = [data.columns.get_loc(k) for k in self.__targets]

        # Denoting history-sequence length
        __limit = self.__n_sequence

        # Hence
        x_matrix = []
        y_matrix = []
        for j in range(data.shape[0] - __limit):
            x_matrix.append(matrix[j:(j + __limit)])
            y_matrix.append(matrix[j + __limit][__indices])

        return np.array(x_matrix), np.array(y_matrix)
