"""Trainer Class

@author: vasudevgupta
"""
import tensorflow as tf
from pathlib import Path

from tf_lightning.callbacks import Callback

class Trainer(object):
    
    ## default args
        
    ## running on only 1 batch for only 1 epoch, No ckpts will be saved
    fast_dev_run = False
        
    start_epoch = 1
    epochs = 10
        
    # if specifying load_dir, it will load_ckpt else it won't load ckpt
    load_dir = ''
    # if loading ckpt, use this ard to check whether all saved objects are restored
    assert_consumed = False
        
    # You need to specify ckpt, if you want to use other save options
    checkpoint = None
        
    # Note: these options will be affected only if you specify checkpoint
    save_only_final_ckpts = False
    save_every_ckpt = False
        
    # by default, all tf_lightning related stuff will be saved in this dir
    lightning_base_dir = 'lightning_stuff'
    ckpt_dir = 'ckpts'
    log_dir= 'logs'
        
    # Arguements related to ckpts
    max_ckpt_to_keep = 3
    keep_checkpoint_every_n_hours = None
        
    # You can override Callback class and customize methods
    callbacks= Callback()
        
    # enable/disable `tf.function`
    enable_function = False
        
    # Only Wandb is supported currently
    logger = 'Wandb'
        
    # define distributed strategy
    strategy = None
        
    # if bool(load_dir):
    #     assert(Save)
    
    default_attrs= [
            'fast_dev_run', 'start_epoch', 'epochs', 'load_dir',
            'assert_consumed', 'checkpoint', 'save_only_final_ckpts',
            'save_every_ckpt', 'saved_ckpts_dir', 'max_ckpt_to_keep',
            'keep_checkpoint_every_n_hours', 'callbacks', 'enable_function',
            'logger', 'strategy',
                    ]
    
    def __init__(self, **kwargs):
        """
        If you won't specify the checkpoint object; nothing will be saved
        If you are specifying the checkpoint, every single checkpoint will be saved
        But if you want to save only final checkpoint, specify `save_every_ckpt` to True
        """
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
        
        if bool(self.strategy):
            tr_dataset = self.strategy.experimental_distribute_dataset(tr_dataset)
            val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)

        self.train(tr_dataset, val_dataset)

    def train(self, tr_dataset, val_dataset):
        
        if self.fast_dev_run:
            print("running on only 1 batch for only 1 epoch, ckpts won't be saved/loaded")
            self.save_only_final_ckpts = False
            self.save_every_ckpt = False
            self.epochs = self.start_epoch
            tr_dataset = tr_dataset.take(1)
            val_dataset = val_dataset.take(1)
            self.load_dir = ''
        
        if self.enable_function:
            self.training_step = tf.function(self.model.training_step)
            
        if self.load_dir: 
            self.load_from_checkpoint(Path(self.lightning_base_dir, self.load_dir),
                                self.assert_consumed)
        
        for epoch in range(self.start_epoch, 1+self.epochs):
            
            self.callbacks.on_epoch_begin(epoch)
            batch_idx= 0
            
            for batch in tr_dataset:
                
                batch_idx += 1
                
                tr_loss= self.training_step(batch, batch_idx)
                val_loss= self.evaluate(val_dataset)
                
                step_metrics= self.callbacks.on_batch_end(tr_loss, val_loss)
            
            if self.save_every_ckpt: self.manager.save()
            
            epoch_metrics= self.callbacks.on_epoch_end(epoch)
        
        if self.save_only_final_ckpts: 
            self.manager.save()
        
        return epoch_metrics
    
    def evaluate(self, val_dataset):

        if self.enable_function:
            self.val_step = tf.function(self.model.validation_step)
        
        batch_idx= 0

        for batch in val_dataset:

            batch_idx += 1

            loss= self.val_step(batch, batch_idx)

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

        for attr in cls.default_attrs:
            parser.add_argument(f'--{attr}', type=type(getattr(cls, attr)), default=getattr(cls, attr))
        
        return parser
        
    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """
        If you wish to over-write some of the args passed through Terminal,
        use this kwargs
        
        Useful especially in case of defining checkpoint object
        """
        args_dictn = {}
        
        for attr in cls.default_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                args_dictn.update({attr: value})
        
        # over-writing over args passed using Terminal
        args_dictn.update(kwargs)
        
        if 'checkpoint' in args_dictn:
            print('==========You are not passing checkpoint object============')
        
        return cls(**args_dictn)

