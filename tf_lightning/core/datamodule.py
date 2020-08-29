# __author__ = 'Vasudev Gupta'

from abc import ABC, abstractmethod


class LightningDataModule(ABC):

    def __init__(self):
        pass

    def prepare_data(self):
        """This method is preferrable to prepare dataset
            - downloading
            - saving
        """
        pass

    def setup(self):
        """This method is preferrable for loading, splitting dataset into train,
        test, val and associating them to this dataloader object
        """
        pass

    @abstractmethod
    def train_dataloader(self):
        """
        returns:
            `tf.data.Dataset` object
        """
        return

    def val_dataloader(self):
        """
        returns:
            `tf.data.Dataset` object
        """
        return

    def test_dataloader(self):
        raise ValueError('Currently NotImplemented')