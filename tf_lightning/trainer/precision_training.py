# __author__ = 'Vasudev Gupta'

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class PrecisionTraining(object):
    """
    Everything related to precision will be handled by lightning :)

    Note:

        Lightning is not changing dtype of model output layer to `float32`

        Since its recommended to use `float32` in the last layer, in case one is training with mixed precision policy.
        You need to change output layer precision to `float32` by yourself
        """
    enable_precision_training = False

    def optimizer_step(self, grads, trainable_variables, batch_idx, optimizer_idx):
        raise NotImplementedError

    def backward(self, loss, trainable_variables, batch_idx, optimizer_idx):
        raise NotImplementedError

    def training_step(batch, batch_idx, optimizer_idx):
        raise NotImplementedError

    # @property
    # def opt_indices(self):
    #     raise NotImplementedError

    # @property
    # def optimizer_0(self):
    #     raise NotImplementedError

    # @property
    # def optimizer_1(self):
    #     raise NotImplementedError

    def __init__(self, policy_name='mixed_float16'):

        if self.enable_precision_training:
            policy = mixed_precision.Policy(policy_name)
            mixed_precision.set_policy(policy)

    def wrap_mixed_precision_optimizer(self):

        for i in self.opt_indices:
            optimizer = getattr(self, f"optimizer_{i}")
            optimizer = mixed_precision.LossScaleOptimizer(
                optimizer, loss_scale='dynamic')
            setattr(self, f"optimizer_{i}", optimizer)

    def _wrapper_precision_train_step(self, batch, batch_idx):

        for optimizer_idx in self.opt_indices:

            result = self.training_step(batch, batch_idx, optimizer_idx)

            scaled_loss = getattr(
                self, f"optimizer_{optimizer_idx}").get_scaled_loss(result['minimize'])

            grads = self.backward(
                scaled_loss, result['trainable_variables'], batch_idx, optimizer_idx)

            grads = getattr(
                self, f"optimizer_{optimizer_idx}").get_unscaled_gradients(grads)

            self.optimizer_step(grads, result['trainable_variables'],
                                batch_idx, optimizer_idx)

        return result
