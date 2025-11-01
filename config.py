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
        Keys
        '''
        self.architecture = 'arc-rnn-lstm'
        self.s3_parameters_key = 's3_parameters.yaml'
        self.arguments_key = f'artefacts/architecture/{self.architecture}/arguments.json'
        

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

        self.pathway_ = os.path.join(self.warehouse, 'arc-rnn-lstm-inference')
        self.points_ = os.path.join(self.pathway_, 'points')
        self.menu_ = os.path.join(self.pathway_, 'menu')


        '''
        Cloud Prefix: Destination
        '''
        self.prefix = 'warehouse/arc-rnn-lstm-inference'


        '''
        Cloud Prefix: Metadata [vis-à-vis Configurations Bucket]
        '''
        self.metadata_ = 'arc-rnn-lstm-inference/external'


        '''
        Cloud Prefix: Artefacts Source
        '''
        self.artefacts_ = f'assets/{self.architecture}'
