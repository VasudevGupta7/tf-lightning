"""Lightning version of TF-2

@author: vasudevgupta
"""

import tensorflow as tf
# from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tf_lightning.callbacks import Callback

class LightningModule(tf.keras.Model):
    
    def __init__(self, save_dir):
        super().__init__()
        
        self.save_dir= save_dir
        
        self.checkpoint= tf.train.Checkpoint(model= kwargs.pop('model', tf.Variable(0.)), 
                                             optimizer= kwargs.pop('optimizer', tf.Variable(0.)))
        
        self.manager= tf.train.CheckpointManager(self.checkpoint, 
                                                 directory=self.save_dir,
                                                 max_to_keep=3,
                                                 keep_checkpoint_every_n_hours=kwargs.pop('keep_checkpoint_every_n_hours', None))
        
        self.callbacks= Callback()
    
    def load_from_checkpoint(self, ckpt, assert_consumed= False):
        # generally: self.manager.latest_checkpoint
        status= self.checkpoint.restore(ckpt)
        if assert_consumed:
            status.assert_consumed()
            logger.info('ckpt_restored')
            
    def train(self,
              tr_dataset,
              val_dataset,
              epochs=10,
              load_dir= None,
              save_every_ckpt= False,
              assert_consumed=False,
              ):
        
        if load_dir: self.load_from_checkpoint(load_dir, assert_consumed)
        
        for epoch in range(1, 1+epochs):
            
            self.callbacks.on_epoch_begin(epoch)
            batch_idx= 0
            
            for batch in tr_dataset:
                
                batch_idx += 1
                
                tr_loss= self.training_step(batch, batch_idx)
                val_loss= self.evaluate(val_dataset)
                
                step_metrics= self.callbacks.on_batch_end(tr_loss, val_loss)
            
            if save_evry_ckpt: self.manager.save()
            
            epoch_metrics= self.callbacks.on_epoch_end(epoch)
        
        if self.save_dir: self.manager.save()
        
        return epoch_metrics
    
    def configure_optimizers(self):
        pass
    
    def evaluate(self, val_dataset):
        
        loss_= 0
        batch_idx= 0
        
        for batch in val_dataset:
            
            batch_idx += 1
            
            loss= self.validation_step(batch, batch_idx)
             
        return loss
        
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
    