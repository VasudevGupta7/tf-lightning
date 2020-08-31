# __author__ = 'Vasudev Gupta'

import os
import wandb
import numpy as np


class WandbLogger(object):
    """
    Parameters:
        logdir: string
            Directory where wandb folder will be created
    Methods:
        log: step=<step number>, metric_name1=<metric_value>, metric_name2=<metric_value>, ......
             metric value can be a scalar, list of numpy array (image)

    """

    def __init__(self,
                 logdir,
                 project_name='lightning-project',
                 config=None):

        local_dirs = os.listdir()
        if logdir not in local_dirs:
            os.makedirs(logdir)

        wandb.init(project=project_name,
                   config=config,
                   dir=logdir)

    def log(self, **kwargs):
        """
        args:
            commit = True will result in incremental logging to wandb
            step = step to associate wandb logging
        """
        commit = kwargs.pop('commit', True)
        step = kwargs.pop('step', None)

        log_dictn = self._get_log_dictn(kwargs)
        wandb.log(log_dictn, commit=commit, step=step)

    def _get_image(self, name, arrays):
        imgs = [wandb.Image(arr) for arr in arrays]
        return {name: imgs}

    def _get_log_dictn(self, dictn):
        log_dictn = {}
        for name, value in dictn.items():
            if np.isscalar(value):
                log_dictn.update({name: value})
            elif isinstance(value, list) and isinstance(value[0], np.ndarray):
                image = self._get_image(name, value)
                log_dictn.update(image)
            else:
                raise ValueError(
                    "Incorrect type, Allowable datatypes are scalar, list of numpy array (image)")
        del dictn
        return log_dictn
