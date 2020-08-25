"""Lightning the tf2 DataLoader part

@author: vasudevgupta
"""
import logging
from abc import ABC, abstractmethod
logger = logging.getLogger(__name__)


class LightningDataModule(ABC):

    def __init__(self):
        pass

    def prepare_data(self):
        """This method is preferrable to prepare dataset
            - downloading
            - saving
            - loading
        """
        pass

    def setup(self):
        """This method if preferrable for splitting dataset into train, test, val
        and associating to this dataloader object
        """
        pass

    @abstractmethod
    def train_dataloader(self):
        """
        This method return `tf.data.Dataset` object for training data
        """
        return

    def val_dataloader(self):
        """
        This method return `tf.data.Dataset` object for val data
        """
        return

    def test_dataloader(self):
        """
        This method return `tf.data.Dataset` object for test data
        """
        return
