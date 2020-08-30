# __author__ = 'Vasudev Gupta'

import tensorflow as tf
from collections.abc import Mapping


class TrainResult(object):
    """[Compulsary]
    Just use this class at the end of your training step..
    Both arguments are compulsary

    Args:
        minimize: pass the loss, you want to minimize
        trainable_variables: pass the variables w.r.t which you want to find gradients
    """

    def __new__(self,
                minimize: tf.Tensor,
                trainable_variables: tf.Tensor,
                log: dict = {}):

        args = {
            'minimize': minimize,
            'trainable_variables': trainable_variables
        }

        args.update(log)

        return args


class EvalResult(object):
    """[Optional]
    Whatever args you will be passing here, will be passed as args in lit-callbacks method
    Args:
        minimize: You can simply pass the loss, if you wish..
    """

    def __new__(self,
                minimize: tf.Tensor,
                log: dict = {}):

        args = {
            'minimize': minimize
        }

        args.update(log)

        return args
