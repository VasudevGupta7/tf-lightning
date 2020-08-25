"""Trainer Class

@author: vasudevgupta
"""
import tensorflow as tf
from pathlib import Path
import logging

from tf_lightning.trainer.training_loop import TrainingLoop
from tf_lightning.trainer.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class Trainer(TrainerConfig, TrainingLoop):

    # if specifying load_dir, it will load_ckpt else it won't load ckpt
    load_dir = ''
    # if loading ckpt, use this ard to check whether all saved objects are restored
    assert_consumed = False

    def __init__(self, **kwargs):
        """
        If you won't specify the checkpoint object; nothing will be saved
        If you are specifying the checkpoint, every single checkpoint will be saved
        But if you want to save only final checkpoint, specify `save_every_ckpt` to True
        """

        # You can change the args specified above
        for attr in kwargs:
            assert(attr in self.default_attrs)
            setattr(self, attr, kwargs[attr])

        super().__init__()

    def fit(self, lightning_module, lightning_data_module):

        # preparing dataset
        lightning_data_module.prepare_data()

        lightning_data_module.setup()

        if not self.enable_distributed_training:
            tr_dataset = lightning_data_module.train_dataloader()
            val_dataset = lightning_data_module.val_dataloader()

        # Inside TrainingLoop
        super().fit(lit_module=lightning_module)

        # testing mode
        if self.fast_dev_run:
            tr_dataset = tr_dataset.take(1)
            val_dataset = val_dataset.take(1)
            self.overwrite_config_for_fast_dev_run()

        if self.load_dir:
            self.load_from_checkpoint(Path(self.lightning_base_dir, self.load_dir),
                                      self.assert_consumed)

        # finally, just training
        history = self.train(tr_dataset, val_dataset)

        return history

    def overwrite_config_for_fast_dev_run(self):
        print("[fast-dev-run mode enabled] :: Model will run on single batch, ckpts won't be saved/loaded")
        self.save_only_final_ckpts = False
        self.save_every_ckpt = False
        self.epochs = self.start_epoch
        self.load_dir = ''

    @classmethod
    def add_argparse_args(cls, parser):

        for attr in cls.default_attrs:
            parser.add_argument(
                f'--{attr}', type=type(getattr(cls, attr)), default=getattr(cls, attr))

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
