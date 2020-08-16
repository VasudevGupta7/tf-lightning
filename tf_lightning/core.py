"""Lightning version of TF-2

@author: vasudevgupta
"""

import tensorflow as tf
# from tensorflow.keras.mixed_precision import experimental as mixed_precision

from callbacks import Callback

class Trainer(object):
    
    def __init__(self, lightning_module, lightning_data_module, **kwargs):
        
        self.epochs= kwargs.get('epochs', 10)
        self.restore_ckpt= kwargs.get('restore_ckpt', False)
        
        
        self.lightning_module= lightning_module
        self.lightning_data_module= lightning_data_module
        
        
    def fit(self):
        
        tr_dataset= self.lightning_data_module.train_dataloader()
        val_dataset= self.lightning_data_module.val_dataloader()
        
        self.lightning_module.train(tr_dataset, val_dataset)

class LightningDataModule(object):
    
    def __init__(self):
        pass
    
    def train_dataloader(self):
        pass
        
    def val_dataloader(self):
        pass
            
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
    
    def restore(self, ckpt, assert_consumed= False):
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
        
        if load_dir: self.restore(load_dir, assert_consumed)
        
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
    
    def configure_optimizers(self)
        pass
    
    def optimizer_step(self, optimizer_idx, batch_idx):
        
        if optimizer_idx == 1:
            tape= 
            trainable_variables= 
            
        elif optimizer_idx == 0:
            tape= 
            trainable_variables= 
            
        gradients= tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
    
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
        
    def validation_step(self, batch, batch_idx, optimizer_idx):
        pass
        
    def test_step(self, batch, batch_idx, optimizer_idx):
        pass
        
    