"""Lightning training for TF-2 with complete flexibilty

@author: vasudevgupta
"""
import tensorflow as tf
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LightningModule(ABC, tf.keras.Model):

    def __init__(self):
        """
        Inherit your model class from this class simply; 
        and you are good to use Trainer class
        """
        optimizer, opt_indices = self._get_optimizer()

        self.opt_indices = opt_indices
        for i in opt_indices:
            setattr(self, f"optimizer_{i}", optimizer[i])
        
        super().__init__(self)
        
    def _get_optimizer(self):
        # called inside integrate_train_step method
        optimizers = self.configure_optimizers()

        if isinstance(optimizers, tf.keras.optimizers.Optimizer):
            opt_indices = [0]

        elif isinstance(optimizers, (tuple, list)):
            opt_indices = list(range(len(optimizers)))

        return optimizers, opt_indices

    def call(self, dataset):
        """[Optional]
        Use it just like you use `call` method of `tf.keras.Model` class
        """
        return

    @abstractmethod
    def configure_optimizers(self):
        """[Necessary]
        You can return either 1 or 2 optimizer depending on your model requirements
        just put comma `,` after optimizer in case you return single optimizer
        """
        return

    def backward(self, loss, trainable_variables, batch_idx, optimizer_idx):
        """[Optional]
        return the gradients of the tensors which were being watched
        Overwrite this method if necessary else you need not really take care of anything
        """
        grads = tf.gradients(loss, trainable_variables)
        return grads

    def optimizer_step(self, grads, trainable_variables, batch_idx, optimizer_idx):
        """[Optional]
        Parameters update stage
        Overwrite this method if necessary else you need not really take care of anything
        """
        if optimizer_idx == 0:
            self.optimizer_0.apply_gradients(zip(grads, trainable_variables))

        elif optimizer_idx == 1:
            self.optimizer_1.apply_gradients(zip(grads, trainable_variables))

    @abstractmethod
    def training_step(self, batch, batch_idx, optimizer_idx):
        """[Necessary]
        Everything is being handled by tf_lightning :)
        - Just define forward propogation in this loop.
        - No need to use `tf.GradientTape`
        - No need to use `tf.function`
        But remember, I am wrapping training_step somewhere in `tf.function`,
        So all rules of working with `tf.function` in defining training loop holds here...

        must specify `minimize`, `trainable_variables` in `TrainResult`
        
        additionally you can specify what to log using `log` attribute of `TrainResult`
        """
        return

    def validation_step(self, batch, batch_idx, optimizer_idx):
        """[Necessary]
        Define the validation step simpy- Only forward propgation will happen

        Everything is being handled by tf_lightning :)
        Remember, I am wrapping this method somewhere in `tf.function`,
        So all rules of working with `tf.function` holds here...

        you may specify `loss` in `EvalResult`

        additionally you can specify what to log using `log` attribute of EvalResult`
        """
        return

    def test_step(self, batch, batch_idx, optimizer_idx):
        return
