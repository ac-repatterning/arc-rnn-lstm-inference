"""
Module config.py
"""
import os


class Config:
    """
    Description
    -----------

    A class for configurations
    """

    def __init__(self) -> None:
        """
        <b>Notes</b><br>
        ------<br>

        Variables denoting a path - including or excluding a filename - have an underscore suffix; this suffix is
        excluded for names such as warehouse, storage, depository, *key, etc.<br><br>
        """

        '''
        Keys:
          The metadata prefix/path is in relation to a cloud configurations bucket.
        '''
        self.architecture = 'arc-rnn-lstm'
        self.s3_parameters_key = 's3_parameters.yaml'
        self.arguments_key = f'architectures/{self.architecture}/arguments.json'
        self.metadata_ = f'architectures/{self.architecture}/inference/external'


        '''
        Project Metadata
        '''
        self.project_tag = 'hydrography'
        self.project_key_name = 'HydrographyProject'


        '''
        Local Paths
        '''
        self.data_ = os.path.join(os.getcwd(), 'data')
        self.warehouse = os.path.join(os.getcwd(), 'warehouse')

        self.pathway_ = self.warehouse
        self.points_ = os.path.join(self.pathway_, 'points')
        self.menu_ = os.path.join(self.pathway_, 'menu')
