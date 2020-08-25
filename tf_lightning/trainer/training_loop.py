"""

@author: vasudevgupta
"""
from pathlib import Path
import tensorflow as tf

from tf_lightning.callbacks import Callback
from tf_lightning.loggers import WandbLogger
from tf_lightning.trainer.checkpointer import Checkpointer
from tf_lightning.trainer.precision_training import PrecisionTraining
from tf_lightning.trainer.distributed_training import DistributedTraining


class TrainingLoop(Checkpointer, PrecisionTraining, DistributedTraining):

    start_epoch = 1
    epochs = 10

    # these arguments are valid only for defalt wandb
    # wandb related arguments
    project_name = 'tf-lightning-project'
    config = None
    sync_tensorboard = False
    save_code = None

    log_dir = 'logs'

    # Related to mixed-precision based training
    policy_name = 'mixed_float16'

    # running on only 1 batch for only 1 epoch, No ckpts will be saved
    fast_dev_run = False

    def __init__(self):

        # You can override Callback class and customize methods
        callbacks = Callback()

        # Wandb is supported by default
        self.lit_logger = WandbLogger(project_name=project_name,
                                      config=config,
                                      log_dir=Path(
                                          lightning_base_dir, log_dir),
                                      sync_tensorboard=sync_tensorboard,
                                      save_code=save_code)

        Checkpointer.__init__()
        PrecisionTraining.__init__(self.policy_name)

    def fit(self, lit_module):

        # adding methods of lightning-module to trainer
        self.integrate_train_step(lit_module)

        if self.enable_precision_training:
            self.wrap_mixed_precision_optimizer()
            self._wrapped_train_step = self._wrapper_precision_train_step

        self.wrap_tf_function()

    def train(self, tr_dataset, val_dataset):

        if bool(self.callbacks):
            self.callbacks.on_train_begin()

        for epoch in range(self.start_epoch, 1+self.epochs):

            if bool(self.callbacks):
                self.callbacks.on_epoch_begin(epoch)

            batch_idx = tf.constant(0)

            for batch in tr_dataset:
                if bool(self.callbacks):
                    self.callbacks.on_batch_begin(batch_idx)

                batch_idx += tf.constant(1)

                tr_result = self.wrapped_train_step(batch, batch_idx)

                val_result = self.evaluate(val_dataset)

                # logging stuff defined in training_step
                if bool(tr_result.log) and (not self.fast_dev_run):
                    self.lit_logger.log(tr_result.log)

                # logging stuff defined in val_step
                if bool(val_result.log) and (not self.fast_dev_run):
                    self.lit_logger.log(val_result.log)

                if bool(self.callbacks):
                    step_metrics = self.callbacks.on_batch_end(
                        batch_idx, tr_result['loss'], val_result['loss'])

            if self.save_every_ckpt:
                self.manager.save()

            if bool(self.callbacks):
                tr_result = self.evaluate(tr_dataset)
                epoch_metrics = self.callbacks.on_epoch_end(
                    epoch, tr_result['loss'], val_result['loss'])

        if bool(self.callbacks):
            self.callbacks.on_train_end()

        if self.save_only_final_ckpts:
            self.manager.save()

        return

    def evaluate(self, val_dataset):
        # called inside train method
        batch_idx = tf.constant(0)

        for batch in val_dataset:

            batch_idx += tf.constant(1)

            val_result = self.val_step(batch, batch_idx, optimizer_idx=0)

        return val_result

    def wrap_tf_function(self):
        # wrapping inside tf.function
        self.wrapped_train_step = tf.function(self._wrapped_train_step)
        self.val_step = tf.function(self.val_step)

    def _wrapper_train_step(self, batch, batch_idx):
        # this method is called inside wrap_tf_function
        """
        This method is simply wrapping everything:
            - forward propogation
            - backward propogation
            - parameters update
        Overwrite this method if necessary else you need not really take care of anything
        """
        for optimizer_idx in self.opt_indices:

            result = self.training_step(batch, batch_idx, optimizer_idx)

            grads = self.backward(
                result['loss'], result['trainable_variables'], batch_idx, optimizer_idx)

            self.optimizer_step(grads, trainable_variables,
                                batch_idx, optimizer_idx)

        return result

    def integrate_train_step(self, lit_module):
        # this will run only once
        # called inside fit method

        self.training_step = lit_module.training_step
        self.val_step = lit_module.val_step

        optimizer, opt_indices = self._get_optimizer(lit_module)
        self.opt_indices = opt_indices
        for i in opt_indices:
            setattr(self, f"optimizer_{i}", optimizer[i])

        self.backward = lit_module.backward

        self.optimizer_step = lit_module.optimizer_step

    def _get_optimizer(self, lit_module):
        # called inside integrate_train_step method
        optimizers = lit_module.configure_optimizers()

        if isinstance(optimizers, tf.keras.optimizers.Optimizer):
            opt_indices = [0]

        elif isinstance(optimizers, (tuple, list)):
            opt_indices = list(range(len(optimizers)))

        return optimizers, opt_indices
