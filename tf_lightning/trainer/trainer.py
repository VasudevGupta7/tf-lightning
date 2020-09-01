# __author__ = 'Vasudev Gupta'

import tensorflow as tf
from pathlib import Path

from tf_lightning.callbacks.lit_callbacks import Callback

from tf_lightning.trainer.training_loop import TrainingLoop
from tf_lightning.trainer.trainer_config import TrainerConfig
from tf_lightning.loggers.wandb import WandbLogger
from tf_lightning.callbacks.checkpointer import Checkpointer
from tf_lightning.trainer.precision_training import PrecisionTraining
from tf_lightning.trainer.distributed_training import DistributedTraining


class Trainer(TrainerConfig, TrainingLoop, PrecisionTraining, Checkpointer):

    def __init__(self, **kwargs):
        """
        If you won't specify the checkpoint object; nothing will be saved
        If you are specifying the checkpoint, every single checkpoint will be saved
        But if you want to save only final checkpoint, specify `save_every_ckpt` to True
        """

        for key in kwargs:
            assert(hasattr(self, key))
            setattr(self, key, kwargs[key])

        if self.enable_loggers and 'loggers' not in kwargs:
            self.loggers = [WandbLogger(project_name=self.project_name,
                                        config=self.litmodule_config,
                                        logdir=self.lightning_base_dir)]

        if not self.callbacks:
            self.enable_callbacks = False

        PrecisionTraining.__init__(self)

    def fit(self, lightning_module, lightning_data_module):

        # preparing dataset
        lightning_data_module.prepare_data()

        lightning_data_module.setup()

        if not self.enable_distributed_training:
            tr_dataset = lightning_data_module.train_dataloader()
            val_dataset = lightning_data_module.val_dataloader()

        TrainingLoop.fit(self, lightning_module)

        # code-testing mode
        if self.fast_dev_run:
            tr_dataset = tr_dataset.take(1)
            val_dataset = val_dataset.take(1)
            self.overwrite_config_for_fast_dev_run()

        if self.load_dir:
            ckpt_name = tf.train.latest_checkpoint(
                Path(self.lightning_base_dir, self.load_dir))
            self.load_from_checkpoint(self.checkpoint, ckpt_name,
                                      self.assert_consumed)

        # finally training
        info = self.train(tr_dataset, val_dataset)

        return info

    def _get_checkpoint_manager(self, checkpoint):
        manager = Checkpointer._get_checkpoint_manager(self, checkpoint)
        return manager

    def overwrite_config_for_fast_dev_run(self):
        print("[fast-dev-run mode enabled] &&& Model will run on single batch, ckpts won't be saved/loaded")
        self.save_only_final_ckpt = False
        self.save_every_ckpt = False
        self.epochs = self.start_epoch + 1
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
        """
        args_dictn = {}

        for attr in cls.default_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                args_dictn.update({attr: value})

        # over-writing over args passed using Terminal
        args_dictn.update(kwargs)

        return cls(**args_dictn)
