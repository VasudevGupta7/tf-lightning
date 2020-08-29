"""Logger class for Wandb and Tensorboard

@author: vasudevgupta
"""
import wandb
import os
import tensorflow as tf
import numpy as np
from matplotlib.figure import Figure
import io

class WandbLogger(object):

    def __init__(self,
                 project_name=None,
                 config=None,
                 log_dir=None,
                 sync_tensorboard=False,
                 save_code=None):

        wandb.init(project=project_name,
                   config=config,
                   dir=log_dir,
                   sync_tensorboard=sync_tensorboard,
                   save_code=save_code)

    def log(self, info, commit=True, step=None):
        wandb.log(info, commit=commit, step=step)

class TBSummary(object):
    """
    Parameters:
        logdir: string
            Directory where all of the events will be written out.
        expt_name: string
            Optional; name for a particular summary, ex. training.
    Attributes:
        writer: tf.summary.FileWriter
            The writer that write `tf_summary` into a tensorflow event file.
    Methods:
        log: step=<step number>, metric_name1=<metric_value>, metric_name2=<metric_value>, ......
             metric value can be a scalar, numpy array (image), matplotlib figure object or a iterable object containing multiple values.
    """
    def __init__(self, logdir, expt_name=""):
        log_path = os.path.join(logdir, expt_name)
        self.writer = tf.summary.create_file_writer(log_path)

    def log(self, **kwargs):
        step = kwargs.pop('step', None)
        for name, value in kwargs.items():
            if hasattr(value, '__len__') and not isinstance(value, np.ndarray):
                self._add_multiple(name, value, step)
            else:
                self._add_single(name, value, step)
    
    def _add_scalar(self, name, value, step=None):
        with self.writer.as_default():
            tf.summary.scalar(name=name, data=value, step=step)
            self.writer.flush()

    def _plot_to_image(self, figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def _add_image(self, name, image, step=None):
        if isinstance(image, Figure):
            image = self._plot_to_image(image)
        elif isinstance(image, np.ndarray):
            image = image
        else:
            raise ValueError("Incorrect Image type. Type must be numpy array or matplotlib figure")
        with self.writer.as_default():
            tf.summary.image(name, image, step=step)
            self.writer.flush()

    def _add_single(self, name, value, step=None):
        if np.isscalar(value):
            self._add_scalar(name, value, step)
        elif isinstance(value, np.ndarray) or isinstance(value, Figure):
            self._add_image(name, value, step)
        else:
            raise ValueError("Incorrect type. Allowable datatypes are scalars, numpy ndarray, matplotlib figures")

    def _add_multiple(self, name, values, step=None):
        for i, value in enumerate(values):
            self._add_single('%s/%d' % (name, i), value, step)