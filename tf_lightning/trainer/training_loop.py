# __author__ = 'Vasudev Gupta'

import tensorflow as tf
from pathlib import Path
from abc import ABC, abstractmethod


class TrainingLoop(ABC):
    """
        Main Class for Training
    """
    @abstractmethod
    def _get_checkpoint_manager(self, checkpoint):
        """Warning: this is just empty shell for code implemented in other class."""

    def fit(self, lit_module):

        self.lit_module = lit_module

        self._integrate_litmodule_optimizer(lit_module)

        self.checkpoint = self.checkpointer()
        self.manager = self._get_checkpoint_manager(self.checkpoint)

        if self.enable_precision_training:
            self._wrap_mixed_precision_optimizer()
            self.train_step = self._wrapper_precision_train_step

        self.wrap_tf_function()

    def train(self, tr_dataset, val_dataset):

        # if bool(self.enable_callbacks):
        #     self.callbacks.on_train_begin()

        batch_idx = tf.constant(0)

        for epoch in range(self.start_epoch, self.epochs):

            # if bool(self.enable_callbacks):
            #     self.callbacks.on_epoch_begin(epoch)

            for batch in tr_dataset:
                # if bool(self.enable_callbacks):
                #     self.callbacks.on_batch_begin(batch_idx)

                batch_idx += tf.constant(1)

                tr_result = self.train_step(batch, batch_idx)
                eval_result = self.evaluate(val_dataset)

                self._batch_end(tr_result, eval_result)

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

        logs = {}

        if self.enable_loggers:

            tr_log = tr_result.pop('log')
            if tr_log:
                for k in tr_log:
                    tr_log[k] = tr_log[k].numpy()

                logs.update(tr_log)
                del tr_log

            if eval_result:
                eval_log = eval_result['log']
                if eval_log:
                    for k in eval_log:
                        eval_log[k] = eval_log[k].numpy()
                    logs.update(eval_log)
                    del eval_log

            for logger in self.loggers:
                logger.log(**logs)

    def _epoch_end(self):
        if self.save_every_ckpt:
            self.manager.save()

    def _training_end(self):
        if self.save_only_final_ckpt:
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
        self.train_step = tf.function(self._wrapped_train_step)
        self.val_step = tf.function(self.validation_step)

    def _wrapped_train_step(self, batch, batch_idx):
        # this method is called inside wrap_tf_function
        """
        This method is simply wrapping everything:
            - forward propogation
            - backward propogation
            - parameters update
        """
        for optimizer_idx in self.opt_indices:

            result = self.training_step(batch, batch_idx, optimizer_idx)

            grads = self.backward(
                result['minimize'], result['trainable_variables'], batch_idx, optimizer_idx)

            self.optimizer_step(grads, result['trainable_variables'],
                                batch_idx, optimizer_idx)

        return result

    def optimizer_step(self, grads, trainable_variables, batch_idx, optimizer_idx):
        return self.lit_module.optimizer_step(grads, trainable_variables, batch_idx, optimizer_idx)

    def backward(self, loss, trainable_variables, batch_idx, optimizer_idx):
        return self.lit_module.backward(loss, trainable_variables, batch_idx, optimizer_idx)

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self.lit_module.training_step(batch, batch_idx, optimizer_idx)

    def validation_step(self, batch, batch_idx, optimizer_idx):
        return self.lit_module.validation_step(batch, batch_idx, optimizer_idx)

    def _integrate_litmodule_optimizer(self, lit_module):

        self.opt_indices = lit_module.opt_indices

        for i in self.opt_indices:
            optim = getattr(lit_module, f'optimizer_{i}')
            setattr(self, f"optimizer_{i}", optim)

    def checkpointer(self):
        return self.lit_module.checkpointer()
