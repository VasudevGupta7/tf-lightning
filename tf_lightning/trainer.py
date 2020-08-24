"""Trainer Class

@author: vasudevgupta
"""
import tensorflow as tf
from pathlib import Path
import logging

from tf_lightning.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)

class Trainer(TrainerConfig):
    
    def __init__(self, **kwargs):
        """
        If you won't specify the checkpoint object; nothing will be saved
        If you are specifying the checkpoint, every single checkpoint will be saved
        But if you want to save only final checkpoint, specify `save_every_ckpt` to True
        """
        
        super().__init__()
        
        ## You can change the args specified above
        for attr in kwargs:
            assert(attr in self.default_attrs) 
            setattr(self, attr, kwargs[attr])
        
        # if no ckpt is given, then nothing will be saved
        if self.checkpoint != None:
            self.save_every_ckpt = True

            if self.save_only_final_ckpts:
                self.save_every_ckpt = False
                
            self.manager= tf.train.CheckpointManager(self.checkpoint, 
                                        directory=Path(self.lightning_base_dir, self.ckpt_dir),
                                        max_to_keep=self.max_ckpt_to_keep,
                                        keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

    def fit(self, lightning_module, lightning_data_module):

        # preparing dataset
        lightning_data_module.prepare_data()

        lightning_data_module.setup()
        tr_dataset = lightning_data_module.train_dataloader()
        val_dataset = lightning_data_module.val_dataloader()

        # defining model
        self.model = lightning_module

        # wrapping in tf.function for good performance
        self.train_step = tf.function(self.model.wrapped_train_step)
        self.val_step = tf.function(self.model.val_step)
        
        # just testing
        if self.fast_dev_run:
            print("Single batch run, ckpts won't be saved/loaded")
            self.save_only_final_ckpts = False
            self.save_every_ckpt = False
            self.epochs = self.start_epoch
            tr_dataset = tr_dataset.take(1)
            val_dataset = val_dataset.take(1)
            self.load_dir = ''

        if self.load_dir: 
            self.load_from_checkpoint(Path(self.lightning_base_dir, self.load_dir),
                                self.assert_consumed)

        # finally, just training
        history = self.train(tr_dataset, val_dataset)

        return history
        
    def train(self, tr_dataset, val_dataset):

        if bool(self.callbacks): 
            self.callbacks.on_train_begin()

        for epoch in range(self.start_epoch, 1+self.epochs):

            if bool(self.callbacks): 
                self.callbacks.on_epoch_begin(epoch)
            
            batch_idx= tf.constant(0)

            for batch in tr_dataset:
                if bool(self.callbacks): 
                    self.callbacks.on_batch_begin(batch_idx)

                batch_idx += tf.constant(1)

                tr_result = self.train_step(batch, batch_idx)

                val_result= self.evaluate(val_dataset)

                # logging stuff defined in training_step
                if bool(tr_result.log) and (not self.fast_dev_run):
                    self.lit_logger.log(tr_result.log)

                # logging stuff defined in val_step
                if bool(val_result.log) and (not self.fast_dev_run):
                    self.lit_logger.log(val_result.log)

                if bool(self.callbacks):
                    step_metrics= self.callbacks.on_batch_end(batch_idx, tr_result['loss'], val_result['loss'])

            if self.save_every_ckpt: self.manager.save()

            # tr_result = self.evaluate(tr_dataset)

            # if bool(self.callbacks):
            #     epoch_metrics= self.callbacks.on_epoch_end(epoch, tr_result['loss'], val_result['loss'])

        if bool(self.callbacks): 
            self.callbacks.on_train_end()

        if self.save_only_final_ckpts: 
            self.manager.save()

        return

    def evaluate(self, val_dataset):
        
        batch_idx= tf.constant(0)

        for batch in val_dataset:

            batch_idx += tf.constant(1)

            val_result = self.val_step(batch, batch_idx, optimizer_idx=0)

        return val_result

    def load_from_checkpoint(self, ckpt, assert_consumed= False):
        # generally: self.manager.latest_checkpoint(ckpt_dir)
        
        status= self.checkpoint.restore(ckpt)
        logger.info('ckpt_restored')
        
        if assert_consumed:
            status.assert_consumed()
            
    def test(self):
        return
    
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

        if 'checkpoint' not in args_dictn:
            print('==========You are not passing checkpoint object============')

        return cls(**args_dictn)

