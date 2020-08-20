"""Trainer Class

@author: vasudevgupta
"""
import tensorflow as tf
from pathlib import Path

from tf_lightning.lightning import LightningModule
from tf_lightning.callbacks import Callback

class Trainer(object):
    
    def __init__(self, **kwargs):
        """
        If you won't specify the checkpoint object; Nothing will be saved
        If you are specifying the checkpoint, every single checkpoint will be saved
        But if you want to save only final checkpoint, specify `save_every_ckpt` to True
        
        """
        ## default args
        
        ## running on only 1 batch for only 1 epoch, No ckpts will be saved
        self.fast_dev_run = False
        
        self.start_epoch = 1
        self.epochs = 10
        
        # if specifying load_dir, it will load_ckpt else it won't load ckpt
        self.load_dir = ''
        # if loading ckpt, use this ard to check whether all saved objects are restored
        self.assert_consumed = False
        
        # You need to specify ckpt, if you want to use other save options
        self.checkpoint = None
        
        # Note: these options will be affected only if you specify checkpoint
        self.save_only_final_ckpts = False
        self.save_every_ckpt = False
        
        # by default, all tf_lightning related stuff will be saved in this dir
        self.lightning_base_dir = 'lightning_stuff'
        self.ckpt_dir = 'ckpts'
        self.log_dir= 'logs'
        
        # Arguements related to ckpts
        self.max_ckpt_to_keep = 3
        self.keep_checkpoint_every_n_hours = None
        
        ## You can override Callback class and customize methods
        self.callbacks= Callback()
        
        # Only Wandb is supported currently
        self.logger = 'Wandb'
        
        self.default_attrs= [
            'fast_dev_run', 'start_epoch', 'epochs', 'load_dir',
            'assert_consumed', 'checkpoint', 'save_only_final_ckpts',
            'save_every_ckpt', 'saved_ckpts_dir', 'max_ckpt_to_keep',
            'keep_checkpoint_every_n_hours', 'callbacks', 'logger'
                       ]
        
        # if bool(load_dir):
        #     assert(Save)
        
        ## You can change the args specified above
        for attr in kwargs:
            assert(attr in self.default_attrs) 
            setattr(self, attr, kwargs[attr])
        
        if self.checkpoint != None:
            self.save_every_ckpt = True

            if self.save_only_final_ckpts:
                self.save_every_ckpt = False
                
            self.manager= tf.train.CheckpointManager(self.checkpoint, 
                                        directory=Path(self.lightning_base_dir, self.ckpt_dir),
                                        max_to_keep=self.max_ckpt_to_keep,
                                        keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)
        
    def fit(self, lightning_module, lightning_data_module):
        
        tr_dataset = lightning_data_module.train_dataloader()
        val_dataset = lightning_data_module.val_dataloader()
        
        self.model = lightning_module 
        
        self.train(tr_dataset, val_dataset)
        
    def train(self, tr_dataset, val_dataset):
        
        if fast_dev_run:
            print('running on only 1 batch for only 1 epoch, No ckpts will be saved')
            self.save_only_final_ckpts = False
            self.save_every_ckpt = False
            self.epochs = self.start_epoch
            tr_dataset = tr_dataset.take(1)
            val_dataset = val_dataset.take(1)
            self.load_dir = ''
        
        if self.load_dir: 
            self.load_from_checkpoint(Path(self.lightning_base_dir, self.load_dir),
                                self.assert_consumed)
        
        for epoch in range(self.start_epoch, 1+self.epochs):
            
            self.callbacks.on_epoch_begin(epoch)
            batch_idx= 0
            
            for batch in tr_dataset:
                
                batch_idx += 1
                
                tr_loss= self.model.training_step(batch, batch_idx)
                val_loss= self.evaluate(val_dataset)
                
                step_metrics= self.callbacks.on_batch_end(tr_loss, val_loss)
            
            if self.save_every_ckpt: self.manager.save()
            
            epoch_metrics= self.callbacks.on_epoch_end(epoch)
        
        if self.save_only_final_ckpts: 
            self.manager.save()
        
        return epoch_metrics
    
    def evaluate(self, val_dataset):

        batch_idx= 0

        for batch in val_dataset:

            batch_idx += 1

            loss= self.model.validation_step(batch, batch_idx)

        return loss

    def load_from_checkpoint(self, ckpt, assert_consumed= False):
        # generally: self.manager.latest_checkpoint(ckpt_dir)
        
        status= self.checkpoint.restore(ckpt)
        if assert_consumed:
            status.assert_consumed()
            logger.info('ckpt_restored')

    def test(self):
        pass
    
    @classmethod
    def add_argparse_args(cls, parser):

        for attr in self.default_attrs:
            parser.add_argument(f'--{attr}', type= int, default= getattr(args, attr))
        
        return parser
        
    @classmethod
    def from_argparse_args(cls, args):
        epochs= args.epochs
        
        for attr in self.default_attrs:
            setattr(args, attr)

        return cls()

