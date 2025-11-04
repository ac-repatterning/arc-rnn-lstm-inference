
import numpy as np
import pandas as pd
import tensorflow as tf

import src.elements.attribute as atr
import src.elements.master as mr
import src.inference.scaling
import src.inference.sequencing


class Estimate:

    def __init__(self, attribute: atr.Attribute):

        self.__attribute = attribute

        # Modelling Arguments
        _, self.__targets, self.__disjoint = self.__get_modelling_arguments()

        # Renaming
        self.__rename = { arg: f'e_{arg}' for arg in self.__targets}

    def __get_modelling_arguments(self):
        """

        :return:
        """

        elements: dict = self.__attribute.modelling
        fields: list = elements.get('fields')
        targets: list = elements.get('targets')

        # The variables present within the [input] fields, but not the targets
        disjoint: list = list(set(fields).difference(set(targets)))

        return fields, targets, disjoint

    def __reconfigure(self, design: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """

        :param design: The design frame, wherein scalable fields have undergone scaling
        :param predictions: The predictions per instance of design; excluding the skipped instances.
        :return:
        """

        # Structuring ahead of inverse transform
        structure = design.copy()[self.__disjoint][-predictions.shape[0]:]
        structure.loc[:, self.__targets] = predictions

        # Inverse transform; of the relevant fields
        frame = src.inference.scaling.Scaling().inverse_transform(
            data=structure, scaling=self.__attribute.scaling)

        return frame.rename(columns=self.__rename, inplace=True)

    def exc(self, model: tf.keras.models.Sequential, master: mr.Master, ):

        x_matrix, _ = src.inference.sequencing.Sequencing(
            modelling=self.__attribute.modelling).exc(blob=master.transforms)

        # Predict
        predictions: np.ndarray = model.predict(x=x_matrix)


        # Reconfiguring
        frame = self.__reconfigure(design=master.transforms, predictions=predictions)

        # Original & Estimates
        __original = master.data.copy()[-predictions.shape[0]:]
        instances = pd.concat([__original.copy().reset_index(drop=True), frame[list(self.__rename.values())]],
                              axis=1)

        return instances
