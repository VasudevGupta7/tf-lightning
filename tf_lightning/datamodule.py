"""Data module class

@author: vasudevgupta
"""

class LightningDataModule(object):
    
    def __init__(self):
        pass
    
    def prepare_data(self):
        """
        Returns
        -------
        None.
        """
        return
    
    def setup(self):
        """
        Returns
        -------
        None.

        """
        return
    
    def train_dataloader(self):
        """
        Returns
        -------
        TYPE
            return `tf.data.Dataset` object
        """
        return
        
    def val_dataloader(self):
        """
        Returns
        -------
        TYPE
            return `tf.data.Dataset` object
        """
        return
    
    def test_dataloader(self):
        """
        Returns
        -------
        TYPE
            return `tf.data.Dataset` object
        """
        return
