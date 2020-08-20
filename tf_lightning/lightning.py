"""Lightning version of TF-2

@author: vasudevgupta
"""
import tensorflow as tf

class LightningModule(tf.keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.optimizer_0, *b = configure_optimizers()
        self.opt_indices = [0]
        
        if bool(b):
            self.optimizer_1 = b[0]
            self.opt_indices = [0, 1]

    def call(self, dataset):
        pass

    def configure_optimizers(self):
        pass

    def get_gradients(self, tape, loss, trainable_variables, batch_idx, optimizer_idx):
        grads = tape.gradient(loss, trainable_variables)
        return grads

    def optimizer_step(self, grads, trainable_variables, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            self.optimizer_0.apply_gradients(zip(grads, trainable_variables))

        elif optimizer_idx == 1:
            self.optimizer_1.apply_gradients(zip(grads, trainable_variables))

    def wrapped_train_step(self, batch, batch_idx):

        for optimizer_idx in self.opt_indices:
            tr_info = self.training_step(batch, batch_idx, optimizer_idx)

            assert('loss' in tr_info)
            tr_loss = tr_info.pop('loss')
            
            assert('trainable_variables' in tr_info)
            trainable_variables = tr_info.pop('trainable_variables')
            
            assert('tape' in tr_info)
            tape = tr_info.pop('tape')

            grads = self.get_gradients(tape, tr_loss, trainable_variables, batch_idx, optimizer_idx)
            self.optimizer_step(grads, trainable_variables, batch_idx, optimizer_idx)

        return tr_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        
        returned stuff must be a dictionary with .....
        `loss`, `tape`, `trainable_variables`, `log` as keys
        """
        pass

    def validation_step(self, batch, batch_idx, optimizer_idx):
        pass

    def test_step(self, batch, batch_idx, optimizer_idx):
        pass
