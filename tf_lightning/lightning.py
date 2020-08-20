"""Lightning version of TF-2

@author: vasudevgupta
"""

import tensorflow as tf
# from tensorflow.keras.mixed_precision import experimental as mixed_precision

class LightningModule(tf.keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__()

    def configure_optimizers(self):
        pass

    def call(self, dataset):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass
        # if optimizer_idx == 1:
        #     tape= 
        #     trainable_variables= 
            
        # elif optimizer_idx == 0:
        #     tape= 
        #     trainable_variables= 
            
        # gradients= tape.gradient(loss, trainable_variables)
        # optimizer.apply_gradients(zip(gradients, trainable_variables))

    def validation_step(self, batch, batch_idx, optimizer_idx):
        pass

    def test_step(self, batch, batch_idx, optimizer_idx):
        pass
