# __author__ = 'Vasudev Gupta'

from pathlib import Path
import tensorflow as tf

from tf_lightning.callbacks.lit_callbacks import Callback
from tf_lightning.loggers.wandb import WandbLogger
from tf_lightning.callbacks.checkpointer import Checkpointer
from tf_lightning.trainer.precision_training import PrecisionTraining
from tf_lightning.trainer.distributed_training import DistributedTraining


class TrainingLoop(Checkpointer, PrecisionTraining, DistributedTraining):

    start_epoch = 1
    epochs = 10

    # wandb related arguments
    project_name = 'tf-lightning-project'
    config = {}

    # Related to mixed-precision based training
    policy_name = 'mixed_float16'

    # running on only 1 batch for only 1 epoch, No ckpts will be saved
    fast_dev_run = False

    enable_callbacks = False

    def __init__(self):

        # You can override Callback class and customize methods
        callbacks = Callback()

        # Wandb is supported by default
        self.lit_logger = WandbLogger(project_name=self.project_name,
                                      config=self.config,
                                      logdir=self.lightning_base_dir)

        Checkpointer.__init__(self)
        PrecisionTraining.__init__(self, self.policy_name)

    def fit(self, lit_module):

        # adding methods of lightning-module to trainer
        self.integrate_train_step(lit_module)

        self.wrapped_train_step = self._wrapped_train_step
        self.val_step = self.validation_step

        if self.enable_precision_training:
            self.wrap_mixed_precision_optimizer()
            self.wrapped_train_step = self._wrapper_precision_train_step

        self.wrap_tf_function()

    def train(self, tr_dataset, val_dataset):

        # if bool(self.enable_callbacks):
        #     self.callbacks.on_train_begin()

        for epoch in range(self.start_epoch, self.epochs):

            # if bool(self.enable_callbacks):
            #     self.callbacks.on_epoch_begin(epoch)

            batch_idx = tf.constant(0)

            for batch in tr_dataset:
                # if bool(self.enable_callbacks):
                #     self.callbacks.on_batch_begin(batch_idx)

                batch_idx += tf.constant(1)

                tr_result = self.wrapped_train_step(batch, batch_idx)
                eval_result = self.evaluate(val_dataset)

                # self._batch_end(tr_result, eval_result)

                # if bool(self.enable_callbacks):
                #     step_metrics = self.callbacks.on_batch_end(
                #         batch_idx, tr_result['loss'], val_result['loss'])

            self._epoch_end()

            # if bool(self.enable_callbacks):
            #     tr_result = self.evaluate(tr_dataset)
            #     epoch_metrics = self.callbacks.on_epoch_end(
            #         epoch, tr_result['loss'], val_result['loss'])

        self._training_end()

        # if bool(self.enable_callbacks):
        #     self.callbacks.on_train_end()

        return

    def _batch_end(self, tr_result, eval_result):

        tr_logs = tr_result.get_log() if bool(tr_result) else None
        eval_logs = eval_result.get_log() if bool(eval_result) else None

        # logging stuff defined in training_step
        if bool(tr_logs) and (not self.fast_dev_run):
            self.lit_logger.log(tr_logs)

            # logging stuff defined in val_step
        if bool(eval_logs) and (not self.fast_dev_run):
            self.lit_logger.log(eval_logs)

    def _epoch_end(self):
        if self.save_every_ckpt:
            self.manager.save()

    def _training_end(self):
        if self.save_only_final_ckpts:
            self.manager.save()

    def evaluate(self, val_dataset):
        # called inside train method
        batch_idx = tf.constant(0)

        for batch in val_dataset:

            batch_idx += tf.constant(1)

            result = self.val_step(batch, batch_idx, optimizer_idx=0)

        return result

    def wrap_tf_function(self):
        # wrapping inside tf.function
        self.wrapped_train_step = tf.function(self.wrapped_train_step)
        self.val_step = tf.function(self.val_step)

    def _wrapped_train_step(self, batch, batch_idx):
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
                result['minimize'], result['trainable_variables'], batch_idx, optimizer_idx)

            self.optimizer_step(grads, result['trainable_variables'],
                                batch_idx, optimizer_idx)

        return result

    def integrate_train_step(self, lit_module):
        # this will run only once
        # called inside fit method

        self.training_step = lit_module.training_step
        self.validation_step = lit_module.validation_step

        self.opt_indices = lit_module.opt_indices
        for i in self.opt_indices:
            optim = getattr(lit_module, f'optimizer_{i}')
            setattr(self, f"optimizer_{i}", optim)

        self.backward = lit_module.backward

        self.optimizer_step = lit_module.optimizer_step
