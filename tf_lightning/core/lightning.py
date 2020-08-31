# __author__ = 'Vasudev Gupta'

import tensorflow as tf
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LightningModule(ABC, tf.keras.Model):
    """
    Inherit your model class from this class simply; and you can use Trainer class simply
    """

    def __init__(self):
        super().__init__()
        optimizer, opt_indices = self._get_optimizer()

        self.opt_indices = opt_indices
        for i in opt_indices:
            setattr(self, f"optimizer_{i}", optimizer[i])

    def call(self, dataset):
        """[Optional]
        Use it just like you use `call` method of `tf.keras.Model` class
        """
        return

    @abstractmethod
    def configure_optimizers(self):
        """[Necessary]
        You can return either 1 or 2 optimizer depending on your model requirements
        """
        return

    def backward(self, loss, trainable_variables, batch_idx, optimizer_idx):
        """[Optional]
        return the gradients of all the trainable_variables which you pass as argument
        Args:
            loss: loss tensor you want to minimize
            trainable_variables: variables w.r.t which you want to calculate your gradients
            batch_idx: No need to chage this
            optimizer_idx: Generally, No need to do anything with this
        """
        grads = tf.gradients(loss, trainable_variables)
        return grads

    def optimizer_step(self, grads, trainable_variables, batch_idx, optimizer_idx):
        """[Optional]
        Parameters update stage
        Args:
            grads: gradients
            trainable_variables: variables w.r.t which you want to calculate your gradients
            batch_idx: No need to chage this
            optimizer_idx: This is based on how you are returning optimizers through `configurble_optimizers`
        """
        if optimizer_idx == 0:
            self.optimizer_0.apply_gradients(zip(grads, trainable_variables))

        elif optimizer_idx == 1:
            self.optimizer_1.apply_gradients(zip(grads, trainable_variables))

    def checkpointer(self):
        """[Optional]
        Define ckpt in this method and return it

        optimizers can be accessed using:
            self.optimizer_0
            self.optimizer_1
        indices depend on how you are returning from `configure_optimizers` method

        Note:
            You need to define the checkpoint incase you want save it.
            By defualt, all epoch-checkpoints are saved; but you can change this setting while initializing `Trainer`

        return:
            `tf.train.Checkpoint` object
        """
        return

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

        additionally you can specify what to log using `log` argument of `TrainResult`
        Everything must be `tf.Tensor`
        """
        return

    def validation_step(self, batch, batch_idx, optimizer_idx):
        """[Necessary]
        Define the validation step simpy- Only forward propgation will happen

        Everything is being handled by tf_lightning :)
        Remember, I am wrapping this method somewhere in `tf.function`,
        So all rules of working with `tf.function` holds here...

        you may specify `loss` in `EvalResult`

        additionally you can specify what to log using `log` argument of EvalResult`
        Everything must be `tf.Tensor`
        """
        return

    def test_step(self, batch, batch_idx, optimizer_idx):
        raise ValueError('Currently NotImplemented')

    def _get_optimizer(self):
        # used in `__init__` method
        optimizers = self.configure_optimizers()

        if isinstance(optimizers, tf.keras.optimizers.Optimizer):
            opt_indices = [0]
        elif isinstance(optimizers, (tuple, list)):
            opt_indices = list(range(len(optimizers)))

        return optimizers, opt_indices
