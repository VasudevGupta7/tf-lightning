# __author__ = 'Vasudev Gupta'

import tensorflow as tf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Checkpointer(object):

    # Note: these options will be affected only if you specify checkpoint
    save_only_final_ckpts = False
    save_every_ckpt = False

    # You need to specify ckpt, if you want to use other save options
    checkpoint = None

    # by default, all tf_lightning related stuff will be saved in this dir
    lightning_base_dir = 'lightning_stuff'
    ckpt_dir = 'ckpts'

    # Arguements related to ckpts
    max_ckpt_to_keep = 3
    keep_checkpoint_every_n_hours = None

    def __init__(self):

        # if no ckpt is given, then nothing will be saved
        if self.checkpoint != None:
            self.save_every_ckpt = True

            if self.save_only_final_ckpts:
                self.save_every_ckpt = False

            self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                      directory=Path(
                                                          self.lightning_base_dir, self.ckpt_dir),
                                                      max_to_keep=self.max_ckpt_to_keep,
                                                      keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

    def load_from_checkpoint(self, ckpt, assert_consumed=False):
        # generally: self.manager.latest_checkpoint(ckpt_dir)
        status = self.checkpoint.restore(ckpt)
        logger.info('ckpt_restored')

        if assert_consumed:
            status.assert_consumed()
